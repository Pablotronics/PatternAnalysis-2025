# train.py
# Project 3 training script — trains U-Net + CAN on HipMRI 2D slices

from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# All artifacts live under the project folder:
MODELS_DIR   = PROJECT_DIR / "models" / "CAN_models"
PREDS_DIR    = PROJECT_DIR / "preds"
PREDS_NII_DIR= PROJECT_DIR / "preds_nii"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDS_DIR.mkdir(parents=True, exist_ok=True)
PREDS_NII_DIR.mkdir(parents=True, exist_ok=True)


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
    ap.add_argument("--batch_size", type=int, default=4, help="This is how many images are processed at once per training step. Decrease is running out of VRAM")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--target_size", type=int, nargs=2, default=[256, 256], help="H W")
    ap.add_argument("--can_dilations", type=int, nargs="+", default=[1,2,4,8,16,32])
    # Route default outputs into this project folder
    ap.add_argument("--out_dir", default=str(MODELS_DIR))
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

    # Histories for plotting
    hist_epochs, hist_train_loss, hist_train_dice, hist_val_dice = [], [], [], []

    PLOTS_DIR = PROJECT_DIR / "plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)




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
        hist_epochs.append(epoch)
        hist_train_loss.append(train_loss)
        hist_train_dice.append(train_dice)
        hist_val_dice.append(val_dice)

    



    # --- Save history to CSV and plot ---
    import csv
    csv_path = PLOTS_DIR / "training_history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_dice", "val_dice"])
        for e, tl, td, vd in zip(hist_epochs, hist_train_loss, hist_train_dice, hist_val_dice):
            w.writerow([e, tl, td, vd])
    print(f"[INFO] Wrote history: {csv_path}")

    plt.figure(figsize=(10,5))
    plt.plot(hist_epochs, hist_train_loss, label="Train Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True, alpha=0.3)
    ax2 = plt.twinx()
    ax2.plot(hist_epochs, hist_train_dice, label="Train Dice", linestyle="--")
    ax2.plot(hist_epochs, hist_val_dice, label="Val Dice", linestyle="-.")
    ax2.set_ylabel("Dice")
    lines_1, labels_1 = plt.gca().get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")
    plt.title("Training Curves (Loss & Dice)")
    fig_path = PLOTS_DIR / "training_curves.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"[INFO] Saved plot: {fig_path}")

    # Final test on best
    print("\nReloading best and testing...")
    state = torch.load(best_ckpt, map_location=device)




    # Final test on best
    print("\nReloading best and testing...")
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    test_dice = evaluate_dice(model, test_dl)
    print(f"TEST Dice (prostate): {test_dice:.4f}")
    print("Best checkpoint:", best_ckpt)
