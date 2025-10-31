# train_mask_generator.py

import os
import shutil
import argparse
import yaml
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm.auto import tqdm
from easydict import EasyDict

try:
    from src.mask_generator_layers import MaskGeneratorNet
except ImportError:
    try:
        from mask_generator_layers import MaskGeneratorNet
    except ImportError:
        print("错误: 无法导入 'MaskGeneratorNet'。")
        print("请确保 'mask_generator_layers.py' 文件在 'src' 目录或当前目录中。")
        exit()

try:
    from src.misc import seed_all, get_logger, get_new_log_dir, inf_iterator, get_checkpoint_path
    from src.common import get_optimizer, get_scheduler
except ImportError:
    print("警告: 无法从 'src.misc' 或 'src.common' 导入工具函数。将使用基本的 print 和简单回退。")
    import datetime
    import random
    import numpy as np

    class SimpleLogger:
        def info(self, msg): print(msg)
        def warning(self, msg): print(f"警告: {msg}")
        def error(self, msg): print(f"错误: {msg}")

    def get_logger(name, log_dir):
        print(f"(注意: 日志未写入文件。无法导入 get_logger)")
        return SimpleLogger()

    def get_new_log_dir(base_dir, prefix='run'):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
        count = 1
        while os.path.exists(log_dir):
             log_dir = os.path.join(base_dir, f"{prefix}_{timestamp}_{count}")
             count += 1
        return log_dir
        
    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def inf_iterator(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    def get_checkpoint_path(ckpt_dir, it=None):
        if it is None: # find latest
            ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt') and f != 'best.pt']
            if not ckpt_files: return None, 0
            latest_it = max(int(f.split('.')[0]) for f in ckpt_files)
            return os.path.join(ckpt_dir, f"{latest_it}.pt"), latest_it
        else:
            path = os.path.join(ckpt_dir, f"{it}.pt")
            if os.path.exists(path): return path, it
            else: return None, 0

    def get_optimizer(cfg, model):
        if cfg.type == 'adam':
            return optim.Adam(
                model.parameters(), lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=(cfg.beta1, cfg.beta2)
            )
        else: raise NotImplementedError('Optimizer not supported: %s' % cfg.type)

    def get_scheduler(cfg, optimizer):
        if cfg.type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=cfg.factor, patience=cfg.patience, min_lr=cfg.min_lr
            )
        elif cfg.type == 'cosine':
             return optim.lr_scheduler.CosineAnnealingLR(
                 optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min
             )
        else: raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def get_edge_length(pos, edge_index):
    row, col = edge_index
    dist = torch.norm(pos[row] - pos[col], p=2, dim=-1).view(-1, 1)
    return dist

def load_data(data_path, dataset_name="", logger=None, config=None):
    log_func = logger.info if logger else print
    log_func_warn = logger.warning if logger else print
    log_func_err = logger.error if logger else print

    log_func(f"正在从 '{data_path}' 加载 {dataset_name} 数据...")
    try:
        with open(data_path, "rb") as f:
            data_list = pickle.load(f)
        if not data_list:
            raise ValueError("加载的数据列表为空。")
        log_func(f"成功加载 {len(data_list)} 个样本。")
    except FileNotFoundError:
        log_func_err(f"错误：找不到数据文件 '{data_path}'。请确保路径正确。")
        raise
    except Exception as e:
        log_func_err(f"加载数据时出错: {e}")
        raise

    max_atom_type_found = 0
    max_edge_type_found = -1
    log_func(f"正在检查 {dataset_name} 数据集中的最大索引...")
    for data in tqdm(data_list, desc=f"检查 {dataset_name} 最大索引"):
        if hasattr(data, 'atom_type') and data.atom_type.numel() > 0:
            current_max_atom = data.atom_type.max().item()
            if current_max_atom > max_atom_type_found:
                max_atom_type_found = current_max_atom
        if hasattr(data, 'edge_type') and data.edge_type.numel() > 0:
            current_max_edge = data.edge_type.max().item()
            if current_max_edge > max_edge_type_found:
                max_edge_type_found = current_max_edge

    log_func(f"--- {dataset_name} 数据集最大索引检查完成 ---")
    log_func(f"    找到的最大原子类型 (max_atom_type): {max_atom_type_found}")
    log_func(f"    找到的最大键类型 (max_edge_type):   {max_edge_type_found}")
    log_func(f"--------------------------------------")

    log_func(f"正在验证 {dataset_name} 数据集中每个样本的索引范围...")
    invalid_samples_indices = []
    if config is None:
        log_func_err("错误：需要 config 对象来进行索引验证。")
        raise ValueError("Config not provided to load_data for validation.")

    atom_limit = config.model.max_atomic_number
    bond_limit = config.model.num_bond_types

    for i, data in enumerate(tqdm(data_list, desc=f"验证 {dataset_name} 索引范围")):
        is_invalid = False
        if hasattr(data, 'atom_type') and data.atom_type.numel() > 0:
            if data.atom_type.min().item() < 0:
                log_func_warn(f"样本 {i} (SMILES: {getattr(data, 'smiles', 'N/A')}) 发现负原子类型: {data.atom_type.min().item()}")
                is_invalid = True
            if data.atom_type.max().item() >= atom_limit:
                 log_func_warn(f"样本 {i} (SMILES: {getattr(data, 'smiles', 'N/A')}) 发现原子类型 >= 配置限制 ({atom_limit}): {data.atom_type.max().item()}")
                 is_invalid = True

        if hasattr(data, 'edge_type') and data.edge_type.numel() > 0:
            if data.edge_type.min().item() < 0:
                log_func_warn(f"样本 {i} (SMILES: {getattr(data, 'smiles', 'N/A')}) 发现负键类型: {data.edge_type.min().item()}")
                is_invalid = True
            if data.edge_type.max().item() >= bond_limit:
                log_func_warn(f"样本 {i} (SMILES: {getattr(data, 'smiles', 'N/A')}) 发现键类型 >= 配置限制 ({bond_limit}): {data.edge_type.max().item()}")
                is_invalid = True

        if is_invalid:
            invalid_samples_indices.append(i)

    if invalid_samples_indices:
        log_func_warn(f"!!! 在 {dataset_name} 数据集中发现 {len(invalid_samples_indices)} 个样本包含无效索引 (负数或超出配置限制) !!!")
        log_func_warn(f"无效样本的索引: {invalid_samples_indices[:20]} {'...' if len(invalid_samples_indices) > 20 else ''}")
        log_func_warn("将过滤掉这些无效样本。")
        valid_data_list = [data for i, data in enumerate(data_list) if i not in invalid_samples_indices]
        log_func_warn(f"过滤后剩余样本数: {len(valid_data_list)}")
        data_list = valid_data_list
    else:
        log_func(f"{dataset_name} 数据集中所有样本索引范围验证通过。")

    log_func("根据 data.pos 计算真实边长...")
    num_missing_pos = 0
    pbar_desc = "计算边长"
    data_iterator = tqdm(data_list, desc=pbar_desc) if logger else data_list
    for i, data in enumerate(data_iterator):
        if not hasattr(data, 'pos') or data.pos is None:
            log_func_err(f"错误：样本 {i} (SMILES: {getattr(data, 'smiles', 'N/A')}) 缺少 'pos' 属性。")
            log_func_err("请确保使用了包含 'pos' 的预处理脚本重新生成数据。")
            num_missing_pos += 1
            if hasattr(data, 'edge_index'):
                data.edge_length = torch.empty((data.edge_index.shape[1], 1), dtype=torch.float32)
            else:
                 data.edge_length = torch.empty((0, 1), dtype=torch.float32)
            continue

        if hasattr(data, 'edge_index') and data.edge_index.numel() > 0:
            data.edge_length = get_edge_length(data.pos, data.edge_index)
        else:
            data.edge_length = torch.empty((0, 1), dtype=torch.float32)

    if num_missing_pos > 0:
        log_func_err(f"严重错误: {num_missing_pos} 个样本缺少 'pos' 属性。请重新运行预处理。")
        exit()
    log_func("真实边长计算完成。")

    return data_list, max_atom_type_found, max_edge_type_found


def main():
    parser = argparse.ArgumentParser(description="训练掩码生成器模型")
    parser.add_argument("config", type=str, help="配置文件的路径 (e.g., configs/train_mask_gen.yml)")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备 (cuda 或 cpu)")
    parser.add_argument("--logdir", type=str, default="./logs_mask_generator", help="日志和检查点保存目录")
    parser.add_argument("--resume_iter", type=int, default=None, help="从指定的迭代次数恢复训练")
    parser.add_argument("--resume_path", type=str, default=None, help="指定检查点目录以恢复训练（优先于 config 目录）")
    args = parser.parse_args()

    resume_from = args.resume_path
    if resume_from:
        config_path = os.path.join(resume_from, "config_mask_gen.yml")
        if not os.path.exists(config_path):
             config_path_alt = os.path.join(os.path.dirname(resume_from), "config_mask_gen.yml")
             if os.path.exists(config_path_alt):
                  config_path = config_path_alt
             else:
                  raise FileNotFoundError(f"在 {resume_from} 或其父目录中找不到 config_mask_gen.yml")
    else:
        config_path = args.config
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"指定的配置文件不存在: {config_path}")

    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))

    config_name = os.path.basename(config_path).split('.')[0]
    seed_all(config.train.seed)

    if resume_from:
        log_dir = resume_from
        print(f"将从日志目录恢复: {log_dir}")
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        try:
             shutil.copyfile(config_path, os.path.join(log_dir, "config_mask_gen.yml"))
        except Exception as e:
             print(f"警告: 无法备份配置文件: {e}")


    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = get_logger("train_mask_gen", log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(f"日志目录: {log_dir}")
    logger.info(f"使用的设备: {args.device}")
    logger.info(f"配置:\n{config}")

    train_data, max_atom_train, max_edge_train = load_data(config.dataset.train, "训练集", logger, config)
    val_data, max_atom_val, max_edge_val = load_data(config.dataset.val, "验证集", logger, config)

    global_max_atom = max(max_atom_train, max_atom_val)
    global_max_edge = max(max_edge_train, max_edge_val)

    logger.info("--- 数据集索引检查结果 ---")
    logger.info(f"数据集中全局最大原子类型: {global_max_atom}")
    logger.info(f"数据集中全局最大键类型:   {global_max_edge}")

    if config.model.max_atomic_number <= global_max_atom:
        logger.error(f"配置错误！config.model.max_atomic_number ({config.model.max_atomic_number}) <= 数据中的最大值 ({global_max_atom})。")
        logger.error(f"请将 max_atomic_number 至少设置为 {global_max_atom + 1}。")
        exit()
    if config.model.num_bond_types <= global_max_edge:
        logger.error(f"配置错误！config.model.num_bond_types ({config.model.num_bond_types}) <= 数据中的最大值 ({global_max_edge})。")
        logger.error(f"请将 num_bond_types 至少设置为 {global_max_edge + 1}。")
        exit()
    logger.info("配置参数 > 数据集最大索引。检查通过。")

    train_loader = DataLoader(
        train_data,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.get('num_workers', 4),
        pin_memory=True if args.device == 'cuda' else False,
        persistent_workers=True if config.train.get('num_workers', 4) > 0 else False
    )
    train_iterator = inf_iterator(train_loader)

    val_loader = DataLoader(
        val_data,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.get('num_workers', 4),
        pin_memory=True if args.device == 'cuda' else False,
        persistent_workers=True if config.train.get('num_workers', 4) > 0 else False
    )
    logger.info(f"用于训练的样本数: {len(train_data)}, 用于验证的样本数: {len(val_data)}")

    logger.info("正在初始化掩码生成器模型...")
    model = MaskGeneratorNet(
        embed_dim=config.model.embed_dim,
        max_atomic_number=config.model.max_atomic_number,
        num_bond_types=config.model.num_bond_types,
        num_convs=config.model.num_convs,
        activation=config.model.activation,
        short_cut=config.model.short_cut,
        concat_hidden=config.model.concat_hidden,
        output_mlp_hidden_dims=config.model.output_mlp_hidden_dims
    ).to(args.device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    optimizer = get_optimizer(config.train.optimizer, model)
    if hasattr(config.train, 'scheduler') and config.train.scheduler.type:
        scheduler = get_scheduler(config.train.scheduler, optimizer)
        logger.info(f"使用调度器: {config.train.scheduler.type}")
    else:
        scheduler = None
        logger.info("未使用学习率调度器。")

    criterion = nn.MSELoss()

    start_iter = 1
    best_val_loss = float('inf')
    if resume_from:
        resume_iter = args.resume_iter
        if resume_iter is None:
             ckpt_path, resume_iter = get_checkpoint_path(ckpt_dir)
             if ckpt_path is None:
                  logger.warning("在恢复目录中未找到检查点，将从头开始训练。")
                  resume_iter = 0
             else:
                  logger.info(f"未指定 resume_iter，将从最新的检查点 {resume_iter} 恢复。")
        else:
             ckpt_path, _ = get_checkpoint_path(ckpt_dir, it=resume_iter)
             if ckpt_path is None:
                  logger.error(f"指定的检查点 {resume_iter} 未找到！将退出。")
                  exit()
             logger.info(f"将从指定的检查点 {resume_iter} 恢复。")

        if ckpt_path:
            try:
                ckpt = torch.load(ckpt_path, map_location=args.device)
                model.load_state_dict(ckpt['model'])
                if 'optimizer' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer'])
                if scheduler and 'scheduler' in ckpt and ckpt['scheduler'] is not None: # Check if scheduler exists in ckpt
                    scheduler.load_state_dict(ckpt['scheduler'])
                start_iter = ckpt.get('iteration', 1) + 1
                best_val_loss = ckpt.get('best_val_loss', float('inf'))
                logger.info(f"成功从迭代 {start_iter - 1} 恢复。当前最佳验证损失: {best_val_loss:.6f}")
            except Exception as e:
                logger.error(f"加载检查点失败: {e}。将从头开始训练。")
                start_iter = 1
                best_val_loss = float('inf')
        else:
            start_iter = 1

    logger.info("开始训练...")
    try:
        for it in range(start_iter, config.train.max_iters + 1):
            model.train()
            optimizer.zero_grad()
            batch = next(train_iterator).to(args.device)

            pred_scores = model(batch)

            target_scores = batch.atom_flexibility_score
            loss = criterion(pred_scores, target_scores)

            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()

            if it % config.train.log_interval == 0:
                logger.info(
                    f"[Train] Iter {it:06d} | Loss {loss.item():.6f} | GradNorm {grad_norm:.4f} | LR {optimizer.param_groups[0]['lr']:.6f}"
                )
                writer.add_scalar("train/loss", loss.item(), it)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], it)
                writer.add_scalar("train/grad_norm", grad_norm, it)
                writer.flush()

            if it % config.train.val_interval == 0 or it == config.train.max_iters:
                sum_val_loss = 0
                sum_n = 0
                with torch.no_grad():
                    model.eval()
                    for val_batch in tqdm(val_loader, desc=f"Validation @ Iter {it}"):
                        val_batch = val_batch.to(args.device)
                        pred_scores_val = model(val_batch)
                        target_scores_val = val_batch.atom_flexibility_score
                        val_loss = criterion(pred_scores_val, target_scores_val)
                        sum_val_loss += val_loss.item() * val_batch.num_graphs
                        sum_n += val_batch.num_graphs

                avg_val_loss = sum_val_loss / sum_n if sum_n > 0 else 0

                if scheduler:
                    if config.train.scheduler.type == "plateau":
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                logger.info(f"[Val] Iter {it:06d} | Avg Loss {avg_val_loss:.6f}")
                writer.add_scalar("val/loss", avg_val_loss, it)
                writer.flush()

                is_best = avg_val_loss < best_val_loss
                if is_best:
                    best_val_loss = avg_val_loss
                    logger.info(f"*** New best validation loss achieved: {best_val_loss:.6f} ***")
                    ckpt_path_best = os.path.join(ckpt_dir, "best.pt")
                    torch.save(
                        {
                            "config": config,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict() if scheduler else None,
                            "iteration": it,
                            "best_val_loss": best_val_loss,
                        },
                        ckpt_path_best,
                    )
                    logger.info(f"最佳模型已保存至: {ckpt_path_best}")

                if it % config.train.save_interval == 0:
                    ckpt_path_iter = os.path.join(ckpt_dir, f"{it}.pt")
                    torch.save(
                        {
                            "config": config,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict() if scheduler else None,
                            "iteration": it,
                            "current_val_loss": avg_val_loss,
                            "best_val_loss": best_val_loss,
                        },
                        ckpt_path_iter,
                    )
                    logger.info(f"检查点已保存至: {ckpt_path_iter}")

    except KeyboardInterrupt:
        logger.info("训练被手动终止。")
    finally:
        writer.close()
        logger.info("训练结束。")


if __name__ == "__main__":
    main()