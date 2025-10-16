import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from rdkit import Chem

# --- 项目相关的导入 ---
from models.epsnet import get_model
from utils.misc import seed_all, get_logger, get_new_log_dir
from utils.chem import set_rdmol_positions, get_best_rmsd


def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="为光谱条件扩散模型进行高性能推理")
    parser.add_argument("ckpt", type=str, help="指向检查点文件的路径")
    # 配置文件现在是可选的，如果检查点目录中没有，则使用此路径
    parser.add_argument("--config", type=str, default=None, help="指向配置文件的路径（可选）")
    parser.add_argument(
        "--data_path", type=str, default="../preprocess/qme14s_test/test.pkl", help="指向测试数据集.pkl文件的路径"
    )
    parser.add_argument("--save_traj", action="store_true", default=False, help="是否保存采样轨迹")
    parser.add_argument("--tag", type=str, default="", help="为输出目录添加标签")
    parser.add_argument("--num_confs", type=int, default=1, help="每个分子生成的构象数量")
    parser.add_argument("--start_idx", type=int, default=0, help="测试集的起始索引")
    parser.add_argument("--end_idx", type=int, default=None, help="测试集的结束索引")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备")
    parser.add_argument("--batch_size", type=int, default=32, help="推理时的批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载使用的工作进程数")

    # --- 采样参数 ---
    parser.add_argument("--n_steps", type=int, default=100, help="采样步数 (推荐用于ODE采样器)")
    parser.add_argument("--w_global", type=float, default=0.5, help="全局梯度的权重")

    args = parser.parse_args()

    # --- 加载检查点和配置 ---
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # 优先使用与检查点在同一目录的配置文件
    config_path_in_ckpt_dir = glob(os.path.join(os.path.dirname(args.ckpt), "*.yml"))
    if config_path_in_ckpt_dir:
        config_path = config_path_in_ckpt_dir[0]
    elif args.config:
        config_path = args.config
    else:
        raise ValueError("错误：在检查点目录中未找到配置文件，并且没有通过 --config 参数指定。")

    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))

    seed_all(config.train.seed if hasattr(config.train, "seed") else 42)

    # --- 日志设置 ---
    log_dir = os.path.dirname(args.ckpt)
    output_dir = get_new_log_dir(log_dir, "inference", tag=args.tag)
    logger = get_logger("inference", output_dir)
    logger.info(f"参数: {args}")

    # --- 数据集和数据加载器 ---
    logger.info(f"正在从以下路径加载测试数据集: {args.data_path}")
    with open(args.data_path, "rb") as f:
        test_data = pickle.load(f)

    end_idx = args.end_idx if args.end_idx is not None else len(test_data)
    test_data_selected = test_data[args.start_idx : end_idx]
    logger.info(f"已选择 {len(test_data_selected)} 个分子进行推理")

    test_loader = DataLoader(
        test_data_selected, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    logger.info("正在初始化模型...")

    # 【核心修改】: 硬编码 training_phase='finetune'
    # 在推理时，我们总是需要加载包含光谱编码器和融合模块的完整模型。
    logger.info("正在以 'finetune' 模式初始化模型以进行推理。")
    model = get_model(config, training_phase="finetune").to(args.device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    results = []
    rmsd_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="正在进行批处理推理"):
            try:
                original_data_list = batch.to_data_list()
                num_samples = args.num_confs

                if num_samples > 1:
                    repeated_list = [d for d in original_data_list for _ in range(num_samples)]
                    inference_batch = Batch.from_data_list(repeated_list).to(args.device)
                else:
                    inference_batch = batch.to(args.device)

                # 运行模型进行构象生成 (推荐使用更快的ODE采样器)
                pos_gen_all, _ = model.langevin_dynamics_sample_ode(
                    batch=inference_batch,
                    n_steps=args.n_steps,
                    w_global=args.w_global,
                )
                pos_gen_all = pos_gen_all.cpu()

                # 将生成的大张量拆分回每个分子的构象列表
                num_nodes_list = [d.num_nodes for d in inference_batch.to_data_list()]
                pos_gen_list = pos_gen_all.split(num_nodes_list)

                current_gen_idx = 0
                for data in original_data_list:
                    confs_for_this_mol = pos_gen_list[current_gen_idx : current_gen_idx + num_samples]
                    current_gen_idx += num_samples

                    conf_rmsds = []
                    for pos_gen_conf in confs_for_this_mol:
                        mol_template = Chem.Mol(data.rdmol)
                        mol_ref = set_rdmol_positions(mol_template, data.pos)
                        mol_gen = set_rdmol_positions(mol_template, pos_gen_conf)
                        rmsd = get_best_rmsd(mol_gen, mol_ref)
                        conf_rmsds.append(rmsd)

                    best_rmsd = min(conf_rmsds)
                    rmsd_list.append(best_rmsd)

                    results.append(
                        {
                            "smiles": data.smiles,
                            "pos_ref": data.pos.clone(),
                            "pos_gen_best": confs_for_this_mol[conf_rmsds.index(best_rmsd)].clone(),
                            "rmsd_all": conf_rmsds,
                            "best_rmsd": best_rmsd,
                        }
                    )

            except Exception as e:
                logger.error(f"处理一个批次时出错: {e}")
                import traceback

                traceback.print_exc()  # 打印详细错误
                continue

    # --- 最终统计与保存 ---
    if rmsd_list:
        mean_rmsd = np.mean(rmsd_list)
        median_rmsd = np.median(rmsd_list)
        logger.info(f"\n{'='*50}\n推理完成\n{'='*50}")
        logger.info(f"已处理 {len(results)} 个分子。")
        logger.info(f"平均 RMSD: {mean_rmsd:.4f} Å | 中位 RMSD: {median_rmsd:.4f} Å")

        results_path = os.path.join(output_dir, "inference_results.pkl")
        with open(results_path, "wb") as f:
            pickle.dump({"results": results, "mean_rmsd": mean_rmsd, "median_rmsd": median_rmsd}, f)
        logger.info(f"结果已保存至: {results_path}")
    else:
        logger.error("没有分子被成功处理！")


if __name__ == "__main__":
    main()
