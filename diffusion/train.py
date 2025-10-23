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
from torch_geometric.loader import DataLoader
import numpy as np # Import numpy for mean calculation

# Import necessary components (adjust paths if needed)
from models.epsnet import get_model
from utils.datasets import ConformationDataset
from utils.misc import seed_all, get_logger, get_new_log_dir, inf_iterator, get_checkpoint_path
from utils.common import get_optimizer, get_scheduler

# --- Imports for RMSD Calculation (Optional, if needed during validation) ---
# from utils.chem import get_best_rmsd, set_rdmol_positions
# from rdkit import Chem
# ------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the new .yml config file (e.g., configs/pp_dihedral.yml)")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='./logs_spectra_dihedral') # Adjusted default log dir
    args = parser.parse_args()

    # --- Configuration and Logging Setup ---
    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
        log_dir = resume_from # Resume to the same directory
        logger = get_logger('train', log_dir) # Get logger for existing dir
        writer = torch.utils.tensorboard.SummaryWriter(log_dir, resume_from_checkpoint=True) # Resume writer
    else:
        config_path = args.config
        with open(config_path, 'r') as f:
            config = EasyDict(yaml.safe_load(f))
        config_name = os.path.basename(config_path).split('.')[0]
        seed_all(config.train.seed if hasattr(config.train, 'seed') else 42) # Use seed from config if available

        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        # Copy code and config only if not resuming
        if os.path.exists(os.path.join(log_dir, 'models')):
             shutil.rmtree(os.path.join(log_dir, 'models'))
        shutil.copytree('./models', os.path.join(log_dir, 'models'))
        shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
        logger = get_logger('train', log_dir) # Get logger for new dir
        writer = torch.utils.tensorboard.SummaryWriter(log_dir) # Create new writer

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load config if resuming (might have changed if resuming from old checkpoint)
    if resume:
         with open(config_path, 'r') as f:
             config = EasyDict(yaml.safe_load(f))

    logger.info(args)
    logger.info(config)

    # --- Datasets and Loaders ---
    logger.info('Loading datasets...')
    try:
        train_set = ConformationDataset(config.dataset.train, transform=None)
        val_set = ConformationDataset(config.dataset.test, transform=None) # Use test path for validation
        if len(train_set) == 0 or len(val_set) == 0:
             raise ValueError("Loaded dataset is empty.")
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}. Please check config paths.")
        exit(1)
    except ValueError as e:
        logger.error(f"Error loading dataset: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading dataset: {e}")
        exit(1)

    train_iterator = inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.get('num_workers', 4), # Add num_workers from config or default
        pin_memory=True if args.device == 'cuda' else False
    ))
    val_loader = DataLoader(
        val_set,
        batch_size=config.train.batch_size, # Use same batch size for consistency
        shuffle=False,
        num_workers=config.train.get('num_workers', 4),
        pin_memory=True if args.device == 'cuda' else False
    )
    logger.info(f'Datasets loaded: Train {len(train_set)}, Validation {len(val_set)}')

    # --- Model, Optimizer, Scheduler ---
    logger.info('Building model...')
    model = get_model(config).to(args.device)
    try:
        model = get_model(config).to(args.device)
        logger.info(f"Model type: {config.model.network}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    except Exception as e:
        logger.error(f"Error building model: {e}")
        exit(1)

    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    start_iter = 1
    best_val_loss = float('inf') # Track best *diffusion* loss

    # --- Resume from Checkpoint ---
    if resume:
        ckpt_path, resume_iter_found = get_checkpoint_path(ckpt_dir, it=args.resume_iter)
        if ckpt_path is not None:
            logger.info('Resuming from checkpoint: %s' % ckpt_path)
            try:
                ckpt = torch.load(ckpt_path, map_location=args.device)
                model.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt['optimizer'])
                # Handle scheduler state loading carefully
                if scheduler and 'scheduler' in ckpt and ckpt['scheduler'] is not None:
                    try:
                        scheduler.load_state_dict(ckpt['scheduler'])
                    except Exception as e_sched:
                        logger.warning(f"Could not load scheduler state: {e_sched}. May reset LR.")
                start_iter = ckpt.get('iteration', resume_iter_found) + 1
                best_val_loss = ckpt.get('best_val_loss', float('inf')) # Load best diffusion loss
                logger.info(f"Resumed from iteration {start_iter - 1}. Best validation diffusion loss: {best_val_loss:.6f}")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}. Starting from scratch.")
                start_iter = 1
                best_val_loss = float('inf')
        else:
            logger.warning("Resume specified but no checkpoint found. Starting from scratch.")
            start_iter = 1
            best_val_loss = float('inf')


    # --- Training Loop ---
    def train(it):
        model.train()
        optimizer.zero_grad()
        try:
            batch = next(train_iterator).to(args.device)
        except StopIteration: # Should not happen with inf_iterator, but as safeguard
            logger.error("Training data iterator exhausted unexpectedly.")
            return

        try:
            # !! Update loss calculation !!
            total_loss, diffusion_loss, loss_dih, kl_separation_loss = model.get_loss(
                batch=batch,
                anneal_power=config.train.get('anneal_power', 2.0), # Use get for safety
            )

            total_loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()

            # Optional: Step scheduler based on type (e.g., Cosine)
            if config.train.scheduler.type == 'cosine':
                scheduler.step()

            # !! Update logging !!
            log_msg = (f"[Train] Iter {it:06d} | TotLoss {total_loss.item():.4f} | DiffLoss {diffusion_loss.item():.4f} | "
                       f"DihLoss {loss_dih.item():.4f} | KLLoss {kl_separation_loss.item():.4f} | "
                       f"Grad {orig_grad_norm:.4f} | LR {optimizer.param_groups[0]['lr']:.6f}")
            logger.info(log_msg)

            # !! Update TensorBoard logging !!
            writer.add_scalar('train/total_loss', total_loss.item(), it)
            writer.add_scalar('train/diffusion_loss', diffusion_loss.item(), it)
            writer.add_scalar('train/dihedral_loss', loss_dih.item(), it)
            writer.add_scalar('train/kl_loss', kl_separation_loss.item(), it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad_norm', orig_grad_norm if orig_grad_norm is not None else 0.0, it) # Handle None case
            writer.flush()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA out of memory during training iteration {it}. Try reducing batch size.")
                # Optional: Graceful exit or try smaller batch?
                raise e # Re-raise error to stop training
            else:
                 logger.error(f"Runtime error during training iteration {it}: {e}")
                 raise e
        except Exception as e:
            logger.error(f"Unexpected error during training iteration {it}: {e}")
            raise e


    # --- Validation Loop ---
    def validate(it):
        # Accumulators for different losses
        sum_total_loss, sum_diffusion_loss, sum_dihedral_loss, sum_kl_loss = 0.0, 0.0, 0.0, 0.0
        sum_n = 0
        loss_values_for_scheduler = [] # Store diffusion losses for scheduler step

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc=f'Validation @ Iter {it}', leave=False):
                batch = batch.to(args.device)
                try:
                    # !! Update loss calculation !!
                    total_loss, diffusion_loss, loss_dih, kl_separation_loss = model.get_loss(
                        batch=batch,
                        anneal_power=config.train.get('anneal_power', 2.0),
                    )

                    # Accumulate all losses for logging
                    batch_size_factor = batch.num_graphs
                    sum_total_loss += total_loss.item() * batch_size_factor
                    sum_diffusion_loss += diffusion_loss.item() * batch_size_factor
                    sum_dihedral_loss += loss_dih.item() * batch_size_factor
                    sum_kl_loss += kl_separation_loss.item() * batch_size_factor
                    sum_n += batch_size_factor

                    # Store diffusion loss for scheduler
                    loss_values_for_scheduler.append(diffusion_loss.item())

                except RuntimeError as e:
                    logger.warning(f"Runtime error during validation batch: {e}. Skipping batch.")
                    continue # Skip this batch
                except Exception as e:
                    logger.warning(f"Unexpected error during validation batch: {e}. Skipping batch.")
                    continue

        # Calculate average losses
        avg_total_loss = sum_total_loss / sum_n if sum_n > 0 else 0.0
        avg_diffusion_loss = sum_diffusion_loss / sum_n if sum_n > 0 else 0.0 # This is the primary metric
        avg_dihedral_loss = sum_dihedral_loss / sum_n if sum_n > 0 else 0.0
        avg_kl_loss = sum_kl_loss / sum_n if sum_n > 0 else 0.0

        # --- Scheduler Step (based on diffusion loss) ---
        if scheduler:
            if config.train.scheduler.type == 'plateau':
                # Plateau scheduler needs a single metric value
                scheduler.step(avg_diffusion_loss)
            elif config.train.scheduler.type != 'cosine': # Cosine steps in train loop
                # Other schedulers might step per epoch/iteration
                 scheduler.step() # Or adapt based on scheduler type

        # --- Logging ---
        logger.info(f'[Validate] Iter {it:06d} | AvgDiffLoss {avg_diffusion_loss:.6f} | AvgDihLoss {avg_dihedral_loss:.6f} | AvgKLLoss {avg_kl_loss:.6f}')
        writer.add_scalar('val/diffusion_loss', avg_diffusion_loss, it)
        writer.add_scalar('val/dihedral_loss', avg_dihedral_loss, it)
        writer.add_scalar('val/kl_loss', avg_kl_loss, it)
        # Log learning rate after potential scheduler step
        writer.add_scalar('val/lr', optimizer.param_groups[0]['lr'], it)
        writer.flush()

        # Return the primary metric for checkpointing decisions
        return avg_diffusion_loss


    # --- Main Loop ---
    logger.info('Start training...')
    try:
        current_val_loss = float('inf') # Initialize current validation loss
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)

            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                current_val_loss = validate(it) # Get avg_diffusion_loss

                # --- Checkpointing (based on diffusion loss) ---
                is_best = current_val_loss < best_val_loss
                if is_best:
                    best_val_loss = current_val_loss
                    logger.info(f'*** Best validation diffusion loss achieved: {best_val_loss:.6f} ***')
                    ckpt_path_best = os.path.join(ckpt_dir, 'best.pt')
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler else None,
                        'iteration': it,
                        'best_val_loss': best_val_loss, # Save the best diffusion loss
                    }, ckpt_path_best)

                # Save regular checkpoint
                if it % config.train.get('save_freq', config.train.val_freq) == 0: # Save at validation freq or specified save_freq
                     ckpt_path_iter = os.path.join(ckpt_dir, f'{it}.pt')
                     torch.save({
                         'config': config,
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict() if scheduler else None,
                         'iteration': it,
                         'current_val_loss': current_val_loss, # Save current diffusion loss
                         'best_val_loss': best_val_loss, # Save best diffusion loss
                     }, ckpt_path_iter)

    except KeyboardInterrupt:
        logger.info('Terminating training...')
    except Exception as e:
         logger.error(f"Training loop terminated due to error: {e}")
         import traceback
         traceback.print_exc() # Print detailed traceback
    finally:
        writer.close()
        logger.info('Training finished.')