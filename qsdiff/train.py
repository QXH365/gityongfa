import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import argparse
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import numpy as np  # --- NEW ---
from scipy.spatial.transform import Rotation as R  # --- NEW ---

# Import all our custom modules
from spectrum_encoder import SpectrumEncoder
from graph_encoder import MolecularGraphEncoder
from diffusion_utils import NoiseScheduler
from denoising_network import DenoisingNetwork 

def get_args():
    parser = argparse.ArgumentParser(description="Train Conformation Generation Diffusion Model")
    parser.add_argument('--dataset_path', type=str, default='qm9s_for_confgf_generation.pkl', help='Path to preprocessed dataset')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for AdamW optimizer')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    
    # Model dimensions
    parser.add_argument('--spec_dim', type=int, default=256, help='Dimension of spectrum embedding')
    parser.add_argument('--node_dim', type=int, default=128, help='Dimension of node features')
    
    # Diffusion parameters
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    
    return parser.parse_args()

# --- NEW: RMSD Calculation Function ---
def calculate_rmsd(pos_pred, pos_ref):
    """
    Calculates the Root Mean Square Deviation (RMSD) between predicted and reference coordinates
    after optimal superposition.

    Args:
        pos_pred (torch.Tensor): Predicted coordinates, shape [N, 3].
        pos_ref (torch.Tensor): Reference (ground truth) coordinates, shape [N, 3].

    Returns:
        float: The RMSD value.
    """
    # Convert to numpy and ensure float64 for precision
    pred = pos_pred.detach().cpu().numpy().astype(np.float64)
    ref = pos_ref.detach().cpu().numpy().astype(np.float64)
    
    # 1. Center the molecules
    pred_centroid = pred.mean(axis=0)
    ref_centroid = ref.mean(axis=0)
    pred_centered = pred - pred_centroid
    ref_centered = ref - ref_centroid
    
    # 2. Find the optimal rotation matrix using Kabsch algorithm (via scipy)
    rotation, _ = R.align_vectors(pred_centered, ref_centered)
    
    # 3. Apply the rotation to the predicted coordinates
    pred_aligned = rotation.apply(pred_centered)
    
    # 4. Calculate the RMSD
    rmsd = np.sqrt(np.mean(np.sum((pred_aligned - ref_centered)**2, axis=1)))
    
    return rmsd

def main():
    args = get_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # --- Load and Prepare Data ---
    print("Loading data...")
    with open(args.dataset_path, 'rb') as f:
        data_list = pickle.load(f)
    
    train_size = int(0.99 * len(data_list))
    train_data, val_data = data_list[:train_size], data_list[train_size:]
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    print(f"Data loaded: {len(train_data)} training samples, {len(val_data)} validation samples.")

    # --- Inspect Data for Embedding Sizes ---
    max_atom_type = max(d.atom_type.max().item() for d in data_list)
    max_edge_type = max(d.edge_type.max().item() for d in data_list)
    num_atom_types, num_edge_types = max_atom_type + 1, max_edge_type + 1

    # --- Initialize Models, Optimizer, and Loss ---
    print("Initializing models...")
    spectrum_encoder = SpectrumEncoder(final_out_dim=args.spec_dim).to(DEVICE)
    graph_encoder = MolecularGraphEncoder(num_atom_types, num_edge_types, node_dim=args.node_dim).to(DEVICE)
    denoising_network = DenoisingNetwork(node_dim=args.node_dim, spec_dim=args.spec_dim).to(DEVICE)
    
    all_params = list(spectrum_encoder.parameters()) + list(graph_encoder.parameters()) + list(denoising_network.parameters())
    
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    loss_fn = nn.MSELoss()
    noise_scheduler = NoiseScheduler(timesteps=args.timesteps)
    noise_scheduler.to(DEVICE)

    best_val_rmsd = float('inf') # --- NEW: Track best RMSD ---

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        spectrum_encoder.train()
        graph_encoder.train()
        denoising_network.train()
        
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch in progress_bar:
            # ... (Training forward pass remains the same)
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            batch_size = batch.num_graphs
            uv_spec = batch.uv_spectrum.reshape(batch_size, -1)
            ir_spec = batch.ir_spectrum.reshape(batch_size, -1)
            raman_spec = batch.raman_spectrum.reshape(batch_size, -1)
            
            spec_condition = spectrum_encoder(uv_spec, ir_spec, raman_spec)
            node_features = graph_encoder(batch.atom_type, batch.edge_index, batch.edge_type)
            
            pos_0 = batch.pos_ref
            t = torch.randint(0, args.timesteps, (batch_size,), device=DEVICE)
            t_broadcast = t.repeat_interleave(batch.ptr.diff(), dim=0)
            noise = torch.randn_like(pos_0)
            noisy_pos = noise_scheduler.add_noise(pos_0, noise, t_broadcast)

            predicted_noise = denoising_network(noisy_pos, node_features, spec_condition, t, batch.batch, batch.edge_index)
            
            loss = loss_fn(predicted_noise, noise)
            
            if torch.isnan(loss):
                print(f"\nWarning: NaN loss detected at epoch {epoch+1}. Skipping batch.")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Step with RMSD Calculation ---
        spectrum_encoder.eval()
        graph_encoder.eval()
        denoising_network.eval()
        
        total_val_rmsd = 0
        num_val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                batch = batch.to(DEVICE)
                batch_size = batch.num_graphs
                uv_spec, ir_spec, raman_spec = [s.reshape(batch_size, -1) for s in [batch.uv_spectrum, batch.ir_spectrum, batch.raman_spectrum]]

                spec_condition = spectrum_encoder(uv_spec, ir_spec, raman_spec)
                node_features = graph_encoder(batch.atom_type, batch.edge_index, batch.edge_type)
                
                pos_ref = batch.pos_ref
                t = torch.randint(0, args.timesteps, (batch_size,), device=DEVICE)
                t_broadcast = t.repeat_interleave(batch.ptr.diff(), dim=0)
                noise = torch.randn_like(pos_ref)
                noisy_pos = noise_scheduler.add_noise(pos_ref, noise, t_broadcast)

                predicted_noise = denoising_network(noisy_pos, node_features, spec_condition, t, batch.batch, batch.edge_index)
                
                # --- NEW: Denoise to get predicted coordinates and calculate RMSD ---
                # Get constants for the denoising formula
                sqrt_alpha_t = noise_scheduler.sqrt_alphas_cumprod.gather(0, t_broadcast).view(-1, 1)
                sqrt_one_minus_alpha_t = noise_scheduler.sqrt_one_minus_alphas_cumprod.gather(0, t_broadcast).view(-1, 1)

                # Predict the original coordinates (x0_pred) from the noisy coordinates and predicted noise
                pos_pred = (noisy_pos - sqrt_one_minus_alpha_t * predicted_noise) / (sqrt_alpha_t + 1e-8) # Add epsilon for stability

                # Iterate through each molecule in the batch to calculate RMSD
                for i in range(batch.num_graphs):
                    start_idx = batch.ptr[i]
                    end_idx = batch.ptr[i+1]
                    
                    # Select predicted and reference coordinates for the i-th molecule
                    mol_pos_pred = pos_pred[start_idx:end_idx]
                    mol_pos_ref = pos_ref[start_idx:end_idx]
                    
                    total_val_rmsd += calculate_rmsd(mol_pos_pred, mol_pos_ref)
                
                num_val_samples += batch.num_graphs

        avg_val_rmsd = total_val_rmsd / num_val_samples if num_val_samples > 0 else 0
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val RMSD: {avg_val_rmsd:.4f}")

        # --- Save checkpoint based on the best (lowest) validation RMSD ---
        if avg_val_rmsd < best_val_rmsd:
            best_val_rmsd = avg_val_rmsd
            checkpoint = {
                'spectrum_encoder_state_dict': spectrum_encoder.state_dict(),
                'graph_encoder_state_dict': graph_encoder.state_dict(),
                'denoising_network_state_dict': denoising_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_rmsd': best_val_rmsd, # Save RMSD in checkpoint
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"🎉 New best model saved with validation RMSD: {best_val_rmsd:.4f}")

if __name__ == '__main__':
    main()