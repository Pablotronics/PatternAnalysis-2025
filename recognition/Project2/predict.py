# predict.py
# Project 3 prediction/evaluation script — loads a CAN checkpoint, evaluates, and optionally saves outputs
import os, glob, argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from modules import UNet2D_CAN

def dice_binary_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    probs = torch.softmax(logits, dim=1)[:, 1]
    pred  = (probs > 0.5).long()
    target = target.long()
    intersection = (pred & target).sum(dim=(1,2)).float()
    union       = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    return ((2*intersection + eps) / (union + eps)).mean().item()

def save_png_triplet(img, gt, pred, out_path_png):
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1); plt.imshow(img.squeeze().cpu().numpy(), cmap='gray'); plt.title('Input'); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(gt.cpu().numpy(), cmap='jet');             plt.title('GT');    plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(pred.cpu().numpy(), cmap='jet');           plt.title('Pred');  plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
    plt.savefig(out_path_png, dpi=120)
    plt.close()

def save_mask_nii(mask_hw: torch.Tensor, out_path_nii: str):
    os.makedirs(os.path.dirname(out_path_nii), exist_ok=True)
    arr = mask_hw.cpu().numpy().astype(np.int16)
    nii = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(nii, out_path_nii)

def get_args():
    ap = argparse.ArgumentParser(description="Evaluate/Infer UNet+CAN on HipMRI (no training)")
    ap.add_argument("--data_root", required=True, help="Path to keras_slices_data folder")
    ap.add_argument("--ckpt", required=False, help="Path to state_dict checkpoint (.pth)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--target_size", type=int, nargs=2, default=[256, 256], help="H W")
    ap.add_argument("--save_pngs", action="store_true")
    ap.add_argument("--save_niis", action="store_true")
    ap.add_argument("--num_samples", type=int, default=9)
    ap.add_argument("--models_dir", default=os.path.join("models", "CAN_models"),
                    help="If --ckpt is not provided, will auto-pick newest best_* from this dir")
    return ap.parse_args()

if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Build test dataset/loader
    img_test = os.path.join(args.data_root, 'keras_slices_test')
    seg_test = os.path.join(args.data_root, 'keras_slices_seg_test')
    test_ds  = HipMRIDataset(img_test, seg_test, target_size=tuple(args.target_size))
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print("Test slices:", len(test_ds))

    # Resolve checkpoint path
    ckpt_path = args.ckpt
    if not ckpt_path:
        pattern = os.path.join(args.models_dir, "unet2d_can_hipmri_best_*.pth")
        cands = glob.glob(pattern)
        if not cands:
            raise FileNotFoundError(f"No checkpoints found at {pattern}. Provide --ckpt explicitly.")
        ckpt_path = max(cands, key=os.path.getctime)
    print("Loading checkpoint:", ckpt_path)

    # Model
    model = UNet2D_CAN(n_classes=2).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Evaluate
    total, n = 0.0, 0
    saved = 0
    for imgs, segs in test_dl:
        imgs, segs = imgs.to(device), segs.to(device)
        with torch.no_grad():
            logits = model(imgs)
            total += dice_binary_from_logits(logits, segs)
            n += 1

            if args.save_pngs or args.save_niis:
                preds = torch.argmax(logits, dim=1).cpu()
                for i in range(min(imgs.size(0), args.num_samples - saved)):
                    name_base = f"sample_{saved:03d}"
                    if args.save_pngs:
                        out_png = os.path.join("preds", f"{name_base}.png")
                        save_png_triplet(imgs[i].cpu(), segs[i].cpu(), preds[i], out_png)
                    if args.save_niis:
                        out_nii = os.path.join("preds_nii", f"{name_base}.nii.gz")
                        save_mask_nii(preds[i], out_nii)
                    saved += 1
                    if saved >= args.num_samples:
                        break

    mean_dice = total / max(1, n)
    print(f"TEST Dice (prostate): {mean_dice:.4f}")
    if args.save_pngs: print("Saved PNG previews to ./preds/")
    if args.save_niis: print("Saved NIfTI masks to ./preds_nii/ (resized space)")
 