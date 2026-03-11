import numpy as np
import torch
import torch.utils.data as data
import os
import pandas as pd
from params import *

# Note: Random seeds are set globally in train_ddpm.py via set_seed()
# Do not set seeds here to avoid overriding the global seed


class XrayCTPADataset(data.Dataset):
    def __init__(self, csv_path, root='.', mode="train", use_augmented_data=False,
                 train_ratio=0.8, val_ratio=0.10, test_ratio=0.10, random_seed=42):
        """
        Dataset that loads CT-Xray pairs from CSV.

        Two modes of operation:
          1. use_augmented_data=True: CSV is already a pre-split file
             (train_augmented.csv, val.csv, or test.csv) produced by
             preprocess_augmented.py. Loads the entire CSV as-is.
          2. use_augmented_data=False: CSV is the full original dataset.
             Splits internally using train_ratio/val_ratio/test_ratio.

        Args:
            csv_path: Path to CSV file
            root: Root directory for data
            mode: One of "train", "val", "test"
            use_augmented_data: If True, CSV is pre-split; load from augmented dirs
            train_ratio: Proportion for training (only used when use_augmented_data=False)
            val_ratio: Proportion for validation (only used when use_augmented_data=False)
            test_ratio: Proportion for testing (only used when use_augmented_data=False)
            random_seed: Seed for reproducible splits (only used when use_augmented_data=False)
        """
        self.root = root
        self.mode = mode
        self.VAE = True

        full_data = pd.read_csv(csv_path)

        if use_augmented_data:
            # Pre-split CSV: load as-is, no internal splitting needed
            self.data = full_data
            self.cts = os.path.join(self.root, CT_PREPROCESSED_AUG_DIR)
            self.xrays = os.path.join(self.root, PX_PREPROCESSED_AUG_DIR)
        else:
            # Original CSV: split internally
            self.cts = os.path.join(self.root, CT_PREPROCESSED_DIR)
            self.xrays = os.path.join(self.root, PX_PREPROCESSED_DIR)

            rng = np.random.RandomState(random_seed)
            n_samples = len(full_data)
            indices = rng.permutation(n_samples)

            train_end = int(train_ratio * n_samples)
            val_end = train_end + int(val_ratio * n_samples)

            if mode == "train":
                split_indices = indices[:train_end]
            elif mode == "val":
                split_indices = indices[train_end:val_end]
            else:  # test
                split_indices = indices[val_end:]

            self.data = full_data.iloc[split_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ct_accession = str(self.data.loc[idx, CT_ACCESSION_COL]).strip()
        cxr_accession = str(self.data.loc[idx, XRAY_ACCESSION_COL]).strip()

        # Load CT (latent or volume)
        ct = np.load(os.path.join(self.cts, ct_accession + '.npy')).astype(np.float32)

        ctout = torch.from_numpy(ct.copy()).float()
        if not self.VAE:
            ctout = ctout.unsqueeze(0)  # (num_slices,H,W) → (1,num_slices,H,W)
        else:
            ctout = ctout.permute(0, 3, 1, 2)  # (4,56,56,128) -> (4,128,56,56)

        # Load matching X-ray 2D image
        xray = torch.from_numpy(np.load(os.path.join(self.xrays, cxr_accession + '.npy'))).float()

        if self.mode in ("train", "val", "test"):
            return {'ct': ctout, 'cxr': xray}
        else:  # infer mode
            return {'ct': ctout, 'cxr': xray, 'ct_accession': ct_accession, 'cxr_accession': cxr_accession}
