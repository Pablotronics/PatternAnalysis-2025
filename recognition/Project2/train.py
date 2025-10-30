# train.py
# Project 3 training script — trains U-Net + CAN on HipMRI 2D slices
import os, time, argparse
import torch
import torch.nn as nn

from dataset import make_loaders
from modules import UNet2D_CAN

def dice_binary_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    probs = torch.softmax(logits, dim=1)[:, 1]  # foreground prob
    pred  = (probs > 0.5).long()
    target = target.long()
    intersection = (pred & target).sum(dim=(1,2)).float()
    union       = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    return ((2*intersection + eps) / (union + eps)).mean().item()

@torch.no_grad()
def evaluate_dice(model: nn.Module, loader) -> float:
    model.eval()
    total = 0.0
    for imgs, segs in loader:
        imgs, segs = imgs.to(device), segs.to(device)
        logits = model(imgs)
        total += dice_binary_from_logits(logits, segs)
    return total / max(1, len(loader))

def get_args():
    ap = argparse.ArgumentParser(description="Train UNet+CAN on HipMRI 2D prostate segmentation")
    ap.add_argument("--data_root", required=True, help="Path to keras_slices_data folder")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--target_size", type=int, nargs=2, default=[256, 256], help="H W")
    ap.add_argument("--can_dilations", type=int, nargs="+", default=[1,2,4,8,16,32])
    ap.add_argument("--out_dir", default=os.path.join("models", "CAN_models"))
    return ap.parse_args()

if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Data
    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = make_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        target_size=tuple(args.target_size),
        num_workers=0
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Model
    model = UNet2D_CAN(n_classes=2, base=64, can_dilations=tuple(args.can_dilations)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Checkpoint path (timestamped)
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    best_ckpt = os.path.join(args.out_dir, f"unet2d_can_hipmri_best_{timestamp}.pth")
    last_ckpt = os.path.join(args.out_dir, "unet2d_can_hipmri_last.pth")
    print("Will save best model to:", best_ckpt)

    # Train
    best_val_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, dice_sum = 0.0, 0.0
        for imgs, segs in train_dl:
            imgs, segs = imgs.to(device), segs.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, segs)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            dice_sum += dice_binary_from_logits(logits, segs)

        train_loss = loss_sum / max(1, len(train_dl))
        train_dice = dice_sum / max(1, len(train_dl))
        val_dice   = evaluate_dice(model, val_dl)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_ckpt)
        torch.save(model.state_dict(), last_ckpt)

        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss {train_loss:.4f} | "
              f"Train Dice {train_dice:.4f} | Val Dice {val_dice:.4f} | Best {best_val_dice:.4f}")

    # Final test on best
    print("\nReloading best and testing...")
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    test_dice = evaluate_dice(model, test_dl)
    print(f"TEST Dice (prostate): {test_dice:.4f}")
    print("Best checkpoint:", best_ckpt)
