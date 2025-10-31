# dataset.py
# Project 3: HipMRI 2D Prostate — dataset & loaders
import os, glob, re
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Prostate label id seen in your masks
DEF_PROSTATE_LABEL = 3

# Match trailing pattern like "004_week_0_slice_0"
_TRAIL_PAT = re.compile(r'(\d+_week_\d+_slice_\d+)$')

def _slice_key_from_stem(stem: str) -> str:
    m = _TRAIL_PAT.search(stem)
    if m:
        return m.group(1)
    if stem.startswith('case_'):
        return stem[len('case_'):]
    if stem.startswith('seg_'):
        return stem[len('seg_'):]
    return stem

def _slice_key(p: Path) -> str:
    return _slice_key_from_stem(p.stem)

def scan_unique_labels(seg_dir: str, k: int = 10):
    """Quickly print unique label ids seen in first k masks."""
    pats = sorted(glob.glob(os.path.join(seg_dir, '**', '*.nii'), recursive=True))[:k]
    uniqs = set()
    for p in pats:
        a = nib.load(p).get_fdata()
        uniqs.update(np.unique(np.rint(a)).astype(int).tolist())
    print('[DEBUG] Unique label ids (first', k, 'masks):', sorted(list(uniqs)))

class HipMRIDataset(Dataset):
    """
    Loads 2D NIfTI slices, robustly pairs images/masks by regex key, normalizes images,
    maps masks to binary prostate-vs-background, and resizes to a fixed size.
    """
    def __init__(
        self,
        img_dir: str,
        seg_dir: str,
        prostate_label: int = DEF_PROSTATE_LABEL,
        target_size: Tuple[int, int] = (256, 256),
    ):
        self.target_size = tuple(target_size)
        img_paths = sorted([Path(p) for p in glob.glob(os.path.join(img_dir, '**', '*.nii'), recursive=True)])
        seg_paths = sorted([Path(p) for p in glob.glob(os.path.join(seg_dir, '**', '*.nii'), recursive=True)])
        if len(img_paths) == 0 or len(seg_paths) == 0:
            raise RuntimeError(f"No .nii files found.\n  img_dir={img_dir}\n  seg_dir={seg_dir}")

        img_map = { _slice_key(p): p for p in img_paths }
        seg_map = { _slice_key(p): p for p in seg_paths }
        common = sorted(set(img_map.keys()) & set(seg_map.keys()))
        if len(common) == 0:
            img_only = sorted(list(set(img_map.keys()) - set(seg_map.keys())))[:10]
            seg_only = sorted(list(set(seg_map.keys()) - set(img_map.keys())))[:10]
            raise RuntimeError(
                "No matching image/seg pairs found.\n"
                f"  Only-in-IMG (first 10): {img_only}\n"
                f"  Only-in-SEG (first 10): {seg_only}\n"
            )

        self.pairs = [(img_map[k], seg_map[k]) for k in common]
        self.prostate_label = prostate_label

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, seg_path = self.pairs[idx]
        img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        seg = nib.load(str(seg_path)).get_fdata()
        seg = np.rint(seg).astype(np.int64)   # integer mask

        # Normalize image (z-score then min-max)
        img = (img - img.mean()) / (img.std() + 1e-8)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Binary mask: prostate vs background
        seg = (seg == self.prostate_label).astype(np.int64)

        # To tensors and resize to fixed size
        img_t = torch.from_numpy(img[None, ...]).float()   # (1,H,W)
        seg_t = torch.from_numpy(seg[None, ...]).float()   # (1,H,W) float for interpolate

        H, W = self.target_size
        img_t = F.interpolate(img_t.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
        seg_t = F.interpolate(seg_t.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
        seg_t = seg_t.squeeze(0).long()  # (H,W) in {0,1}
        return img_t, seg_t

def make_loaders(
    data_root: str,
    batch_size: int = 4,
    target_size: Tuple[int, int] = (256, 256),
    num_workers: int = 0,
):
    img_train = os.path.join(data_root, 'keras_slices_train')
    seg_train = os.path.join(data_root, 'keras_slices_seg_train')
    img_val   = os.path.join(data_root, 'keras_slices_validate')
    seg_val   = os.path.join(data_root, 'keras_slices_seg_validate')
    img_test  = os.path.join(data_root, 'keras_slices_test')
    seg_test  = os.path.join(data_root, 'keras_slices_seg_test')

    train_ds = HipMRIDataset(img_train, seg_train, target_size=target_size)
    val_ds   = HipMRIDataset(img_val,   seg_val,   target_size=target_size)
    test_ds  = HipMRIDataset(img_test,  seg_test,  target_size=target_size)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl
