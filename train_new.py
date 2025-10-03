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

from models.epsnet import get_model
from utils.datasets import ConformationDataset
from utils.misc import *
from utils.common import get_optimizer, get_scheduler

# --- Imports for RMSD Calculation ---
from utils.chem import get_best_rmsd, set_rdmol_positions
from rdkit import Chem
# -----------------------------
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the new .yml config file (e.g., configs/qm9_spectra.yml)")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='./logs_spectra')
    args = parser.parse_args()

    # --- Configuration and Logging Setup (Unchanged) ---
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
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        if os.path.exists(os.path.join(log_dir, 'models')):
             shutil.rmtree(os.path.join(log_dir, 'models'))
        shutil.copytree('./models', os.path.join(log_dir, 'models'))
        shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)

    # --- 🔄 Datasets and Loaders (MODIFIED SECTION) ---
    logger.info('Loading datasets from specified files...')

    # 检查配置文件中是否提供了测试/验证集的路径
    if not hasattr(config.dataset, 'test') or config.dataset.test is None:
        logger.error("Configuration error: 'config.dataset.test' path is not specified.")
        logger.error("Please provide a separate file for the test/validation set in your YAML config.")
        exit(1) # 终止程序

    # 1. 从 config.dataset.train 路径加载训练集
    logger.info(f"Loading training set from: {config.dataset.train}")
    train_set = ConformationDataset(config.dataset.train, transform=None)
    
    # 2. 从 config.dataset.test 路径加载验证集
    logger.info(f"Loading validation set from: {config.dataset.test}")
    val_set = ConformationDataset(config.dataset.test, transform=None)
    
    # 3. 为加载好的数据集创建 DataLoaders
    train_iterator = inf_iterator(DataLoader(train_set, config.train.batch_size, shuffle=True))
    val_loader = DataLoader(val_set, batch_size=config.train.batch_size, shuffle=False) # 验证集 batch_size 可与训练集一致或设为1
    
    logger.info(f'✅ Datasets loaded successfully.')
    logger.info(f'   - Train size: {len(train_set)}')
    logger.info(f'   - Validation size: {len(val_set)}')
    # --- END OF MODIFIED SECTION ---

    # --- Model, Optimizer, Scheduler (Unchanged) ---
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    start_iter = 1

    # --- Resume from Checkpoint (Unchanged) ---
    if resume and args.resume_iter is not None:
        ckpt_path, start_iter = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)
        logger.info('Resuming from: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    # --- Training Loop (Unchanged) ---
    def train(it):
        model.train()
        optimizer.zero_grad()
        batch = next(train_iterator).to(args.device)
        loss, loss_global, loss_local = model.get_loss(
            batch=batch, anneal_power=config.train.anneal_power,
        )
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        if config.train.scheduler.type == 'cosine':
            scheduler.step()
        logger.info('[Train] Iter %05d | Loss %.2f | Loss(Global) %.2f | Loss(Local) %.2f | Grad %.2f | LR %.6f' % (
            it, loss.item(), loss_global.item(), loss_local.item(), orig_grad_norm, optimizer.param_groups[0]['lr'],
        ))
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/loss_global', loss_global, it)
        writer.add_scalar('train/loss_local', loss_local, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad_norm', orig_grad_norm, it)
        writer.flush()


    # --- Validation Loop (Unchanged) ---
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
                break
        avg_loss = sum_loss / sum_n if sum_n > 0 else 0

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type != 'cosine':
            scheduler.step()

        
        logger.info(f'[Validate] Iter {it:05d} | Loss {avg_loss:.6f}')
        writer.add_scalar('val/loss', avg_loss, it)
        writer.flush()
        return avg_loss


    # --- Main Loop (Unchanged) ---
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
        logger.info('Terminating...')