import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score
import numpy as np
import click
from pathlib import Path
import json
from tqdm import tqdm

from dataset import create_dataloaders
from model import SideChainPredictor

def calculate_metrics(preds, targets):
    """
    计算支链级别的评估指标。
    
    Args:
        preds (torch.Tensor): 模型的sigmoid输出 [num_branches, vocab_size]
        targets (torch.Tensor): 真实标签 [num_branches, vocab_size]
        
    Returns:
        tuple: (exact_match_ratio, f1_macro)
    """
    if preds.shape[0] == 0:
        return 0.0, 0.0
        
    # 将概率转换为0/1预测
    preds_binary = (preds > 0.5).int()
    
    # 1. 精确匹配率 (Exact Match Ratio)
    # 逐行比较，当整行完全相同时才算匹配
    exact_matches = torch.all(preds_binary == targets, dim=1).float().sum()
    exact_match_ratio = (exact_matches / targets.shape[0]).item()
    
    # 2. F1分数 (macro-averaged)
    # 使用sklearn计算，需要转到CPU和numpy
    targets_np = targets.cpu().numpy()
    preds_binary_np = preds_binary.cpu().numpy()
    
    # 'macro'为每个标签计算指标，然后取未加权的平均值。
    f1_macro = f1_score(targets_np, preds_binary_np, average='macro', zero_division=0)
    
    return exact_match_ratio, f1_macro


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for data in tqdm(dataloader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        
        logits = model(data)
        
        # 只在挂载点上计算损失和指标
        attachment_mask = data.attachment_mask
        active_logits = logits[attachment_mask]
        active_targets = data.y[attachment_mask]
        
        if active_logits.shape[0] == 0: continue

        loss = criterion(active_logits, active_targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        
        all_preds.append(torch.sigmoid(active_logits).detach())
        all_targets.append(active_targets.detach())

    avg_loss = total_loss / len(dataloader.dataset)
    
    # 计算整个epoch的指标
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    exact_match, f1 = calculate_metrics(all_preds, all_targets)
    
    return avg_loss, exact_match, f1


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            data = data.to(device)
            logits = model(data)

            attachment_mask = data.attachment_mask
            active_logits = logits[attachment_mask]
            active_targets = data.y[attachment_mask]
            
            if active_logits.shape[0] == 0: continue

            loss = criterion(active_logits, active_targets)
            total_loss += loss.item() * data.num_graphs
            
            all_preds.append(torch.sigmoid(active_logits))
            all_targets.append(active_targets)

    avg_loss = total_loss / len(dataloader.dataset)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    exact_match, f1 = calculate_metrics(all_preds, all_targets)
    
    return avg_loss, exact_match, f1


@click.command()
@click.option("--data_dir", type=click.Path(exists=True, path_type=Path), default='./side_chain_processed_data', help="Directory containing processed data.")
@click.option("--save_dir", type=click.Path(path_type=Path), default="./saved_models", help="Directory to save models and results.")
@click.option("--epochs", type=int, default=50, help="Number of training epochs.")
@click.option("--batch_size", type=int, default=32, help="Batch size.")
@click.option("--lr", type=float, default=1e-3, help="Learning rate.")
@click.option("--seed", type=int, default=42, help="Random seed.")
def main(data_dir, save_dir, epochs, batch_size, lr, seed):
    """Main training and evaluation script."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Create DataLoaders
    train_loader, val_loader, _, atom_vocab_size = create_dataloaders(data_dir, batch_size, seed)
    
    # 加载bond_dict以确定词汇表大小
    with open(data_dir / "bond_dict.json", 'r') as f:
        bond_dict = json.load(f)
    bond_vocab_size = len(bond_dict)

    # 2. Initialize Model, Optimizer, Criterion
    model = SideChainPredictor(
        atom_vocab_size=atom_vocab_size, 
        bond_vocab_size=bond_vocab_size
    ).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
    
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() # 适合多标签分类

    # 3. Training Loop
    best_val_f1 = -1
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_match, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_match, val_f1 = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}/{epochs}:")
        print(f"  Train -> Loss: {train_loss:.4f}, Exact Match: {train_match:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}, Exact Match: {val_match:.4f}, F1: {val_f1:.4f}")
        
        epoch_results = {
            'epoch': epoch,
            'train_loss': train_loss, 'train_exact_match': train_match, 'train_f1': train_f1,
            'val_loss': val_loss, 'val_exact_match': val_match, 'val_f1': val_f1
        }
        history.append(epoch_results)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  New best model saved with F1: {best_val_f1:.4f}")

    # Save training history
    with open(save_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=4)
        
    print("\nTraining complete!")
    print(f"Best validation F1 score: {best_val_f1:.4f}")
    print(f"Best model saved to {save_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()