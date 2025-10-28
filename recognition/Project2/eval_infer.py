import os, glob, re
from pathlib import Path
from typing import Tuple
import argparse

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --------- Args ---------
def get_args():
    ap = argparse.ArgumentParser(description="Evaluate trained UNet on HipMRI test set (no training).")
    ap.add_argument("--data_root", required=True, help=r"Path to keras_slices_data (e.g. C:\...\keras_slices_data)")
    ap.add_argument("--ckpt", default="models/unet2d_hipmri_best.pth", help="Path to state_dict checkpoint")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--target_size", type=int, nargs=2, default=[256, 256], help="H W resize")
    ap.add_argument("--save_pngs", action="store_true", help="Save a few prediction PNGs to preds/")
    ap.add_argument("--save_niis", action="store_true", help="Save predicted masks as NIfTI in preds_nii/")
    ap.add_argument("--num_samples", type=int, default=6, help="How many samples to save if --save_pngs/--save_niis")
    return ap.parse_args()

# --------- Constants / pairing helpers ---------
DEF_PROSTATE_LABEL = 3  # from your sanity check: labels were [0,1,2,3], prostate=3
_TRAIL_PAT = re.compile(r"(\d+_week_\d+_slice_\d+)$")

def _slice_key_from_stem(stem: str) -> str:
    m = _TRAIL_PAT.search(stem)
    if m: return m.group(1)
    if stem.startswith("case_"): return stem[len("case_"):]
    if stem.startswith("seg_"):  return stem[len("seg_"):]
    return stem

def _slice_key(p: Path) -> str:
    return _slice_key_from_stem(p.stem)

# --------- Dataset with 256x256 resize ---------
class HipMRIDataset(Dataset):
    def __init__(self, img_dir: str, seg_dir: str, prostate_label: int = DEF_PROSTATE_LABEL,
                 target_size=(256,256)):
        self.target_size = tuple(target_size)
        img_paths = sorted([Path(p) for p in glob.glob(os.path.join(img_dir, '**', '*.nii'), recursive=True)])
        seg_paths = sorted([Path(p) for p in glob.glob(os.path.join(seg_dir, '**', '*.nii'), recursive=True)])

        if len(img_paths) == 0 or len(seg_paths) == 0:
            raise RuntimeError(f"No .nii files found.\n  img_dir={img_dir}\n  seg_dir={seg_dir}")

        img_map = { _slice_key(p): p for p in img_paths }
        seg_map = { _slice_key(p): p for p in seg_paths }
        common = sorted(set(img_map.keys()) & set(seg_map.keys()))
        if len(common) == 0:
            raise RuntimeError("No matching image/seg pairs found. Check naming/regex.")

        self.pairs = [(img_map[k], seg_map[k]) for k in common]
        self.prostate_label = prostate_label

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, seg_path = self.pairs[idx]
        img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        seg = nib.load(str(seg_path)).get_fdata()
        seg = np.rint(seg).astype(np.int64)

        # normalize
        img = (img - img.mean()) / (img.std() + 1e-8)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # binary prostate vs background
        seg = (seg == self.prostate_label).astype(np.int64)

        # to tensors
        img_t = torch.from_numpy(img[None, ...]).float()    # (1,H,W)
        seg_t = torch.from_numpy(seg[None, ...]).float()    # (1,H,W) float for interpolate

        # resize to fixed size
        H, W = self.target_size
        img_t = F.interpolate(img_t.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
        seg_t = F.interpolate(seg_t.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
        seg_t = seg_t.squeeze(0).long()  # (H,W) in {0,1}
        return img_t, seg_t, img_path.name  # include name for saving preds

# --------- UNet (must match training) ---------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet2D(nn.Module):
    def __init__(self, n_classes=2, base=64):
        super().__init__()
        self.inc   = DoubleConv(1, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base,   base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))
        self.up1   = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.conv1 = DoubleConv(base*8, base*4)
        self.up2   = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv2 = DoubleConv(base*4, base*2)
        self.up3   = nn.ConvTranspose2d(base*2, base,   2, stride=2)
        self.conv3 = DoubleConv(base*2, base)
        self.outc  = nn.Conv2d(base, n_classes, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x  = self.up1(x4)
        x  = torch.cat([x, x3], dim=1); x = self.conv1(x)
        x  = self.up2(x)
        x  = torch.cat([x, x2], dim=1); x = self.conv2(x)
        x  = self.up3(x)
        x  = torch.cat([x, x1], dim=1); x = self.conv3(x)
        return self.outc(x)

# --------- Metrics ---------
def dice_binary_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    probs = torch.softmax(logits, dim=1)[:, 1]  # foreground prob
    pred  = (probs > 0.5).long()
    target = target.long()
    intersection = (pred & target).sum(dim=(1,2)).float()
    union       = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    return ((2*intersection + eps) / (union + eps)).mean().item()

# --------- Save helpers ---------
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
    """
    Saves a (H,W) mask as NIfTI in resized space (identity affine).
    Note: if you need original spacing/orientation, you’d map back using the source affine/shape.
    """
    os.makedirs(os.path.dirname(out_path_nii), exist_ok=True)
    arr = mask_hw.cpu().numpy().astype(np.int16)
    nii = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(nii, out_path_nii)

# --------- Main ---------
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Paths
    img_test_dir = os.path.join(args.data_root, 'keras_slices_test')
    seg_test_dir = os.path.join(args.data_root, 'keras_slices_seg_test')

    # Data
    test_ds  = HipMRIDataset(img_test_dir, seg_test_dir, prostate_label=DEF_PROSTATE_LABEL,
                             target_size=tuple(args.target_size))
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test slices: {len(test_ds)}")

    # Model
    model = UNet2D(n_classes=2).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Evaluate
    total_dice, batches = 0.0, 0
    saved = 0
    for imgs, segs, names in test_dl:
        imgs, segs = imgs.to(device), segs.to(device)
        with torch.no_grad():
            logits = model(imgs)
            total_dice += dice_binary_from_logits(logits, segs)
            batches += 1

            if args.save_pngs or args.save_niis:
                preds = torch.argmax(logits, dim=1).cpu()
                for i in range(min(imgs.size(0), args.num_samples - saved)):
                    name_base = Path(names[i]).stem  # e.g., case_004_week_0_slice_0
                    if args.save_pngs:
                        out_png = os.path.join("preds", f"{name_base}.png")
                        save_png_triplet(imgs[i].cpu(), segs[i].cpu(), preds[i], out_png)
                    if args.save_niis:
                        out_nii = os.path.join("preds_nii", f"{name_base}.nii.gz")
                        save_mask_nii(preds[i], out_nii)
                    saved += 1
                    if saved >= args.num_samples:
                        break
        if saved >= args.num_samples:
            # no need to save more previews
            pass

    mean_dice = total_dice / max(1, batches)
    print(f"TEST Dice (prostate): {mean_dice:.4f}")
    if args.save_pngs:
        print("Saved PNG previews in ./preds/")
    if args.save_niis:
        print("Saved NIfTI masks in ./preds_nii/ (resized space, identity affine)")

if __name__ == "__main__":
    main()
