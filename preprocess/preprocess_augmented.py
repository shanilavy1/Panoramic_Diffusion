"""
Offline augmentation pipeline: Split data first, then generate N augmented
versions of ONLY the training pairs from raw DICOMs.

Pipeline:
  1. Read original CSV (all pairs)
  2. Split into train/val/test using same seed and ratios as training config
  3. Augment only train pairs (original + N augmented versions)
  4. Preprocess all (augment -> normalize -> resize -> VAE encode)
  5. Save 3 separate CSVs: train_augmented.csv, val.csv, test.csv

Augmentations (MONAI transforms, applied to raw data before preprocessing):
  - X-ray (2D): RandAffined (±3° rotation, ±5px translation),
                RandScaleIntensityd (±5% brightness),
                RandAdjustContrastd (gamma 0.95-1.05)
  - CT (3D):    RandAffined (±3° Z-axis rotation, ±5 voxel H/W translation),
                RandShiftIntensityd (±20 HU offset)
  - NO flips (dental anatomy is left-right asymmetric)
  - NO Gaussian noise

Usage:
    python preprocess/preprocess_augmented.py
"""

import os
import sys
import numpy as np
import pandas as pd
import pydicom
import torch

# Add project root to path so we can import from preprocess/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocess.preprocess_ctpa import (
    dicom_load_scan, dicom_get_pixels_hu, preprocess_ctpa, encode_ctpa
)
from preprocess.preprocess_xray import preprocess_xray
from diffusers import AutoencoderKL
from params import DEVICE

from monai.transforms import (
    Compose,
    RandAffined,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    EnsureTyped,
)


# ============================================================
# MONAI augmentation pipelines
# ============================================================

def create_xray_augmentation():
    """
    MONAI augmentation pipeline for raw 2D X-ray images.
    Expects input dict with key 'image' of shape [1, H, W] (channel-first).

    All transforms use prob=1.0 because we decide externally whether to
    augment (aug_idx > 0) or not (aug_idx == 0).
    """
    return Compose([
        EnsureTyped(keys=["image"], dtype=torch.float32),
        RandAffined(
            keys=["image"],
            prob=1.0,
            rotate_range=(0.052,),          # ±3° in radians for 2D
            translate_range=(5, 5),          # ±5 pixels
            mode="bilinear",
            padding_mode="reflection",
        ),
        RandScaleIntensityd(
            keys=["image"],
            factors=0.05,                    # ±5% brightness
            prob=1.0,
        ),
    ])


def create_ct_augmentation():
    """
    MONAI augmentation pipeline for raw 3D CT volumes in Hounsfield Units.
    Expects input dict with key 'image' of shape [1, Z, H, W] (channel-first).
    """
    return Compose([
        EnsureTyped(keys=["image"], dtype=torch.float32),
        RandAffined(
            keys=["image"],
            prob=1.0,
            rotate_range=(0.052, 0.052, 0.052),  # ±3° all axis
            translate_range=(0, 5, 5),         # ±5 voxels in H/W, no Z shift
            mode="bilinear",
            padding_mode="border",
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=20.0,                      # ±20 HU offset
            prob=1.0,
        ),
    ])


# ============================================================
# Helpers
# ============================================================

def load_raw_pair(ct_accession, xray_accession, ct_src_path, xray_src_path):
    """Load raw DICOM data for one CT-Xray pair. Returns None on failure."""
    ct_scan_dir = os.path.join(ct_src_path, ct_accession)
    if not os.path.isdir(ct_scan_dir):
        print(f"  WARNING: CT dir not found: {ct_scan_dir}, skipping")
        return None
    ct_dicom_files = [os.path.join(ct_scan_dir, f)
                      for f in os.listdir(ct_scan_dir) if f.lower().endswith(".dcm")]
    if not ct_dicom_files:
        print(f"  WARNING: No DICOM files in {ct_scan_dir}, skipping")
        return None
    slices, attr = dicom_load_scan(ct_dicom_files)
    ct_hu = dicom_get_pixels_hu(slices)

    xray_scan_dir = os.path.join(xray_src_path, xray_accession)
    if not os.path.isdir(xray_scan_dir):
        print(f"  WARNING: X-ray dir not found: {xray_scan_dir}, skipping")
        return None
    dcm_files = [f for f in os.listdir(xray_scan_dir) if f.lower().endswith('.dcm')]
    if not dcm_files:
        print(f"  WARNING: No DICOM files in {xray_scan_dir}, skipping")
        return None
    ds = pydicom.dcmread(os.path.join(xray_scan_dir, dcm_files[0]))
    xray_raw = ds.pixel_array.astype(np.float32)
    if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == 'MONOCHROME1':
        xray_raw = xray_raw.max() - xray_raw
        print(f"  Inverted MONOCHROME1 -> MONOCHROME2")
    return ct_hu, attr, xray_raw


def preprocess_and_save_pair(ct_hu, attr, xray_raw, ct_filename, xray_filename,
                             ct_dst_path, xray_dst_path, vae):
    """Preprocess one CT-Xray pair and save as .npy files."""
    xray_processed = preprocess_xray(xray_raw)
    np.save(os.path.join(xray_dst_path, xray_filename + '.npy'), xray_processed)
    ct_preprocessed = preprocess_ctpa(ct_hu, attr)
    ct_latent = encode_ctpa(ct_preprocessed, vae)
    np.save(os.path.join(ct_dst_path, ct_filename + '.npy'), ct_latent)


# ============================================================
# Main pipeline
# ============================================================

def preprocess_augmented_directory(
    csv_path, ct_src_path, xray_src_path, ct_dst_path, xray_dst_path,
    output_csv_dir, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10,
    num_augmentations=10, seed=42,
):
    """
    Split data first, then augment only the training pairs.

    Produces 3 CSVs:
      - train_augmented.csv: train pairs (original + N augmented versions each)
      - val.csv: val pairs (original only)
      - test.csv: test pairs (original only)

    All .npy files go to the same ct_dst_path / xray_dst_path directories.
    """
    os.makedirs(ct_dst_path, exist_ok=True)
    os.makedirs(xray_dst_path, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)

    # ---- Step 1: Split FIRST using same logic as training ----
    df = pd.read_csv(csv_path)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(df))
    train_end = int(train_ratio * len(df))
    val_end = train_end + int(val_ratio * len(df))

    train_df = df.iloc[indices[:train_end]].reset_index(drop=True)
    val_df = df.iloc[indices[train_end:val_end]].reset_index(drop=True)
    test_df = df.iloc[indices[val_end:]].reset_index(drop=True)

    print(f"Dataset split (seed={seed}):")
    print(f"  Train: {len(train_df)} pairs (will produce {len(train_df) * (num_augmentations + 1)} samples)")
    print(f"  Val:   {len(val_df)} pairs (original only)")
    print(f"  Test:  {len(test_df)} pairs (original only)\n")

    # ---- Step 2: Load VAE ----
    print("Loading VAE model...")
    vae = AutoencoderKL.from_single_file(
        "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors")
    vae.to(DEVICE, dtype=torch.float32)
    print("VAE loaded.\n")

    xray_aug = create_xray_augmentation()
    ct_aug = create_ct_augmentation()
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- Step 3: TRAIN (original + augmented) ----
    print(f"{'='*60}")
    print(f"Processing TRAIN set ({len(train_df)} pairs x {num_augmentations + 1})")
    print(f"{'='*60}\n")

    train_rows = []
    for i, row in train_df.iterrows():
        ct_acc = str(row['CT_accession_number']).strip()
        xray_acc = str(row['XRAY_Accession_number']).strip()
        print(f"[Train {i+1}/{len(train_df)}] CT={ct_acc}, X-ray={xray_acc}")

        raw = load_raw_pair(ct_acc, xray_acc, ct_src_path, xray_src_path)
        if raw is None:
            continue
        ct_hu_orig, attr, xray_raw_orig = raw
        print(f"  CT: {ct_hu_orig.shape}, X-ray: {xray_raw_orig.shape}")

        for aug_idx in range(num_augmentations + 1):
            if aug_idx == 0:
                xray_raw = xray_raw_orig.copy()
                ct_hu = ct_hu_orig.copy()
                print(f"  [aug 0] Original")
            else:
                # Augment X-ray with MONAI
                xray_data = {"image": xray_raw_orig.copy()[np.newaxis]}
                xray_result = xray_aug(xray_data)
                xray_raw = xray_result["image"].squeeze(0)
                if isinstance(xray_raw, torch.Tensor):
                    xray_raw = xray_raw.numpy()

                # Augment CT with MONAI (independent random state)
                ct_data = {"image": ct_hu_orig.copy().astype(np.float32)[np.newaxis]}
                ct_result = ct_aug(ct_data)
                ct_hu = ct_result["image"].squeeze(0)
                if isinstance(ct_hu, torch.Tensor):
                    ct_hu = ct_hu.numpy()
                ct_hu = ct_hu.astype(np.int16)
                print(f"  [aug {aug_idx}] MONAI augmented")

            suffix = f"_{aug_idx}"
            ct_fn = f"{ct_acc}{suffix}"
            xray_fn = f"{xray_acc}{suffix}"
            preprocess_and_save_pair(ct_hu, attr, xray_raw, ct_fn, xray_fn,
                                     ct_dst_path, xray_dst_path, vae)
            train_rows.append({
                'CT_accession_number': ct_fn,
                'XRAY_Accession_number': xray_fn,
            })
        print()

    # ---- Step 4: VAL (original only) ----
    print(f"{'='*60}")
    print(f"Processing VAL set ({len(val_df)} pairs, original only)")
    print(f"{'='*60}\n")

    val_rows = []
    for i, row in val_df.iterrows():
        ct_acc = str(row['CT_accession_number']).strip()
        xray_acc = str(row['XRAY_Accession_number']).strip()
        print(f"[Val {i+1}/{len(val_df)}] CT={ct_acc}, X-ray={xray_acc}")

        raw = load_raw_pair(ct_acc, xray_acc, ct_src_path, xray_src_path)
        if raw is None:
            continue
        ct_hu, attr, xray_raw = raw
        ct_fn = f"{ct_acc}_0"
        xray_fn = f"{xray_acc}_0"
        preprocess_and_save_pair(ct_hu, attr, xray_raw, ct_fn, xray_fn,
                                 ct_dst_path, xray_dst_path, vae)
        val_rows.append({
            'CT_accession_number': ct_fn,
            'XRAY_Accession_number': xray_fn,
        })
        print()

    # ---- Step 5: TEST (original only) ----
    print(f"{'='*60}")
    print(f"Processing TEST set ({len(test_df)} pairs, original only)")
    print(f"{'='*60}\n")

    test_rows = []
    for i, row in test_df.iterrows():
        ct_acc = str(row['CT_accession_number']).strip()
        xray_acc = str(row['XRAY_Accession_number']).strip()
        print(f"[Test {i+1}/{len(test_df)}] CT={ct_acc}, X-ray={xray_acc}")

        raw = load_raw_pair(ct_acc, xray_acc, ct_src_path, xray_src_path)
        if raw is None:
            continue
        ct_hu, attr, xray_raw = raw
        ct_fn = f"{ct_acc}_0"
        xray_fn = f"{xray_acc}_0"
        preprocess_and_save_pair(ct_hu, attr, xray_raw, ct_fn, xray_fn,
                                 ct_dst_path, xray_dst_path, vae)
        test_rows.append({
            'CT_accession_number': ct_fn,
            'XRAY_Accession_number': xray_fn,
        })
        print()

    # ---- Step 6: Save 3 separate CSVs ----
    train_csv = os.path.join(output_csv_dir, 'train_augmented.csv')
    val_csv = os.path.join(output_csv_dir, 'val.csv')
    test_csv = os.path.join(output_csv_dir, 'test.csv')

    pd.DataFrame(train_rows).to_csv(train_csv, index=False)
    pd.DataFrame(val_rows).to_csv(val_csv, index=False)
    pd.DataFrame(test_rows).to_csv(test_csv, index=False)

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Train CSV: {train_csv} ({len(train_rows)} samples)")
    print(f"  Val CSV:   {val_csv} ({len(val_rows)} samples)")
    print(f"  Test CSV:  {test_csv} ({len(test_rows)} samples)")
    print(f"  CT .npy:   {ct_dst_path}")
    print(f"  X-ray .npy: {xray_dst_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # ---- CONFIGURE THESE PATHS ----
    BASE_DIR = "/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/CT_PX_dataset_diffusion"

    preprocess_augmented_directory(
        csv_path=os.path.join(BASE_DIR, "Facial_PX_dataset.csv"),
        ct_src_path=os.path.join(BASE_DIR, "facial_bones_2025_2020_spine"),
        xray_src_path=os.path.join(BASE_DIR, "PX_2025_2020_spine"),
        ct_dst_path=os.path.join(BASE_DIR, "CT_preprocessed_aug"),
        xray_dst_path=os.path.join(BASE_DIR, "PX_preprocessed_aug"),
        output_csv_dir=BASE_DIR,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        num_augmentations=10,
        seed=42,
    )
