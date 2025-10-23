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

# --- Project-specific Imports ---
from models.epsnet import get_model
from utils.misc import seed_all, get_logger, get_new_log_dir
# ✅ CORRECTNESS: Using the proven RMSD calculation functions from your training script
from utils.chem import set_rdmol_positions, get_best_rmsd

def main():
    # --- Argument Parsing (Standard) ---
    parser = argparse.ArgumentParser(description='High-Performance Inference for GeoDiff')
    parser.add_argument('ckpt', type=str, help='Path to the checkpoint file')
    parser.add_argument('--config', type=str, default='configs/pp.yml', help='Path to config file (if not in ckpt dir)')
    parser.add_argument('--data_path', type=str, default='../../qsdiff_v3/src/qme14s_all/test_data.pkl', help='Path to test dataset pkl file')
    parser.add_argument('--save_traj', action='store_true', default=False, help='Whether to save the sampling trajectory')
    parser.add_argument('--tag', type=str, default='', help='Tag for the output directory')
    parser.add_argument('--num_confs', type=int, default=1, help='Number of conformations to generate per molecule')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for the test set')
    parser.add_argument('--end_idx', type=int, default=256000, help='End index for the test set')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # --- Sampling Parameters (Standard) ---
    parser.add_argument('--n_steps', type=int, default=50, help='Number of sampling steps')
    parser.add_argument('--eta', type=float, default=1.0, help='Sampling eta parameter')
    parser.add_argument('--w_global', type=float, default=0.5, help='Weight for global gradients')
    
    args = parser.parse_args()
    
    # --- Load Checkpoint and Config (Standard) ---
    ckpt = torch.load(args.ckpt, map_location='cpu')
    config_path = args.config or glob(os.path.join(os.path.dirname(args.ckpt), '*.yml'))[0]
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    
    seed_all(config.train.seed if hasattr(config.train, 'seed') else 42)
    
    # --- Logging Setup (Standard) ---
    log_dir = os.path.dirname(args.ckpt)
    output_dir = get_new_log_dir(log_dir, 'inference_batched', tag=args.tag)
    logger = get_logger('inference', output_dir)
    logger.info(f"Arguments: {args}")

    # --- Dataset and DataLoader (Batched for Speed) ---
    logger.info(f"Loading test dataset from: {args.data_path}")
    with open(args.data_path, 'rb') as f:
        test_data = pickle.load(f)
    
    end_idx = args.end_idx if args.end_idx is not None else len(test_data)
    test_data_selected = test_data[args.start_idx:end_idx]
    logger.info(f"Selected {len(test_data_selected)} molecules for inference")

    test_loader = DataLoader(
        test_data_selected,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # --- Model Initialization (Standard) ---
    logger.info("Initializing model...")
    model = get_model(config.model).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    results = []
    rmsd_list = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running batched inference"):
            try:
                original_data_list = batch.to_data_list()
                num_samples = args.num_confs

                # SPEED: Expand batch for parallel sampling if num_samples > 1
                if num_samples > 1:
                    repeated_list = [d for d in original_data_list for _ in range(num_samples)]
                    inference_batch = Batch.from_data_list(repeated_list).to(args.device)
                else:
                    inference_batch = batch.to(args.device)
                
                # Run the model on the entire batch

                #pos_gen_all, _ = model.langevin_dynamics_sample(
                #    batch=inference_batch,
                #    w_global=args.w_global
                #)
                pos_gen_all, pos_traj = model.langevin_dynamics_sample_ode(
                    batch=batch,
                    n_steps=args.n_steps,
                    w_global=args.w_global,
                )
                pos_gen_all = pos_gen_all.cpu()
                
                # Split the large tensor of generated positions back into a list of tensors
                num_nodes_list = [d.num_nodes for d in inference_batch.to_data_list()]
                pos_gen_list = pos_gen_all.split(num_nodes_list)

                current_gen_idx = 0
                for data in original_data_list:
                    # Extract the N generated conformations for the current molecule
                    confs_for_this_mol = pos_gen_list[current_gen_idx : current_gen_idx + num_samples]
                    current_gen_idx += num_samples
                    
                    conf_rmsds = []
                    # ✅ CORRECTNESS: This loop now uses your proven RMSD logic
                    for pos_gen_conf in confs_for_this_mol:
                        # Create a clean copy of the molecule template to avoid side effects
                        mol_template = Chem.Mol(data.rdmol)
                        
                        # Set positions for the reference and generated conformers
                        mol_ref = set_rdmol_positions(mol_template, data.pos)
                        mol_gen = set_rdmol_positions(mol_template, pos_gen_conf)

                        # Calculate RMSD using the trusted function
                        rmsd = get_best_rmsd(mol_gen, mol_ref)
                        conf_rmsds.append(rmsd)
                    
                    best_rmsd = min(conf_rmsds)
                    rmsd_list.append(best_rmsd)
                    
                    # Store results (standard)
                    results.append({
                        'smiles': data.smiles,
                        'pos_ref': data.pos.clone(),
                        'pos_gen_best': confs_for_this_mol[conf_rmsds.index(best_rmsd)].clone(),
                        'rmsd_all': conf_rmsds,
                        'best_rmsd': best_rmsd
                    })
                    print('rmsd:',conf_rmsds)

            except Exception as e:
                logger.error(f"Error processing a batch: {e}")
                continue

    # --- Final Statistics and Saving (Standard) ---
    if rmsd_list:
        mean_rmsd = np.mean(rmsd_list)
        median_rmsd = np.median(rmsd_list)
        logger.info(f"\n{'='*50}\nINFERENCE COMPLETE\n{'='*50}")
        logger.info(f"Processed {len(results)} molecules.")
        logger.info(f"Mean RMSD: {mean_rmsd:.4f} Å | Median RMSD: {median_rmsd:.4f} Å")
        
        results_path = os.path.join(output_dir, 'inference_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump({'results': results, 'mean_rmsd': mean_rmsd, 'median_rmsd': median_rmsd}, f)
        logger.info(f"Results saved to: {results_path}")
    else:
        logger.error("No molecules were successfully processed!")

if __name__ == '__main__':
    main()