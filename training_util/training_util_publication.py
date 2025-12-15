import numpy as np
import torch
from torch.cuda import amp
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data_util.calculate_stats import compute_r2, compute_mae

class MOFDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pc, extra, target, row_idx = self.samples[idx]
        return pc, extra, target, row_idx

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    scaler = amp.GradScaler()

    all_preds = []
    all_targets = []
    all_indicies = []

    progress_bar = tqdm(train_loader, desc="Train", leave=False)

    for pc, extra_features, target, idx in progress_bar:
        pc = pc.to(device)
        extra_features = extra_features.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        with amp.autocast():
            pred, _ = model(pc, extra_features)
            loss = loss_fn(pred, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        all_preds.append(pred.detach().cpu().numpy().reshape(-1))
        all_targets.append(target.cpu().numpy().reshape(-1))
        all_indicies.append(idx.cpu().numpy().reshape(-1))

        avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix(loss=avg_loss)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_indicies = np.concatenate(all_indicies)

    r2 = compute_r2(all_targets, all_preds)
    mae = compute_mae(all_targets, all_preds)
    print(f"  Train R2: {r2:.5f}")
    print(f"  Train MAE: {mae:.5f}")
    
    return total_loss / len(train_loader), all_preds, all_targets, all_indicies

def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_indicies = []

    with torch.no_grad():
        for pc, extra_features, target, idx in tqdm(loader, desc="Evaluating", leave=False):
            pc = pc.to(device)
            extra_features = extra_features.to(device)
            target = target.to(device)

            pred, _ = model(pc, extra_features)
            loss = loss_fn(pred, target)
           
            running_loss += loss.item() * pc.shape[0]
            all_preds.append(pred.cpu().numpy().reshape(-1))
            all_targets.append(target.cpu().numpy().reshape(-1))
            all_indicies.append(idx.cpu().numpy().reshape(-1))

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_indicies = np.concatenate(all_indicies, axis=0)

    r2 = compute_r2(all_targets, all_preds)
    mae = compute_mae(all_targets, all_preds)
    print(f"  Val R2: {r2:.5f}")
    print(f"  Val MAE: {mae:.5f}")

    return epoch_loss, all_preds, all_targets, all_indicies