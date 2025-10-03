# train_new.py

import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader

# 确保 get_model 函数能被正确调用
from models.epsnet import get_model
from utils.datasets import ConformationDataset
from utils.misc import *
from utils.common import get_optimizer, get_scheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --- 简化后的命令行参数 ---
    parser.add_argument('config', type=str, help="配置文件的路径 (e.g., configs/pp_v4.yml)")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='./logs_spectra')
    args = parser.parse_args()

    # --- 配置和日志设置 (保持不变) ---
    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path).split('.')[0]
    seed_all(config.train.seed)

    if resume:
        log_dir = resume_from
    else:
        # 日志目录名不再包含 training_phase
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        if not os.path.exists(os.path.join(log_dir, 'models')):
             shutil.copytree('./models', os.path.join(log_dir, 'models'))
        shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)

    # --- 数据集加载 (保持不变) ---
    logger.info('正在加载数据集...')
    train_set = ConformationDataset(config.dataset.train)
    val_set = ConformationDataset(config.dataset.val)
    train_iterator = inf_iterator(DataLoader(train_set, config.train.batch_size, shuffle=True))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)
    logger.info(f'数据集加载成功: 训练集 {len(train_set)} | 验证集 {len(val_set)}')

    # --- 模型、优化器、调度器 (简化) ---
    logger.info('正在构建模型...')
    # 直接调用 get_model，不再需要传递 training_phase
    model = get_model(config).to(args.device)
    
    # 移除了所有关于加载预训练权重和冻结参数的逻辑
    
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    start_iter = 1
    
    # --- 从检查点恢复 (保持不变) ---
    if resume and args.resume_iter is not None:
        ckpt_path, start_iter = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)
        logger.info('正在从检查点恢复: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    # --- 训练与验证循环 (保持不变) ---
    def train(it):
        model.train()
        optimizer.zero_grad()
        batch = next(train_iterator).to(args.device)
        loss, loss_global, loss_local = model.get_loss(batch=batch, anneal_power=config.train.anneal_power)
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        
        logger.info('[训练] Iter %05d | 总损失 %.2f | 全局损失 %.2f | 局部损失 %.2f | 梯度范数 %.2f | 学习率 %.6f' % (
            it, loss.item(), loss_global.item(), loss_local.item(), orig_grad_norm, optimizer.param_groups[0]['lr'],
        ))
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/loss_global', loss_global, it)
        writer.add_scalar('train/loss_local', loss_local, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad_norm', orig_grad_norm, it)
        writer.flush()


    def validate(it):
        sum_loss, sum_n = 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validation'):
                batch = batch.to(args.device)
                loss, _, _ = model.get_loss(
                    batch=batch, anneal_power=config.train.anneal_power,
                )
                sum_loss += loss.item() * batch.num_graphs
                sum_n += batch.num_graphs
        avg_loss = sum_loss / sum_n if sum_n > 0 else 0

        # ... (调度器和日志记录保持不变) ...
        return avg_loss

    # --- 主循环 (保持不变) ---
    try:
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                avg_val_loss = validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_val_loss,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('终止训练...')