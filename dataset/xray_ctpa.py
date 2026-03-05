import numpy as np
import torch
import random
import torch.utils.data as data
import os
import os.path
import pandas as pd
from params import *

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = False

class XrayCTPADataset(data.Dataset):
    def __init__(self, root='.', csv_path=None, mode="train", augmentation=False,
                 train_ratio=0.8, val_ratio=0.10, test_ratio=0.10, random_seed=42):
        """
        Args:
            root: Root directory for data
            csv_path: Path to CSV file with data info
            mode: One of "train", "val", or "test"
            augmentation: Whether to apply data augmentation
            train_ratio: Proportion of data for training (default 0.8)
            val_ratio: Proportion of data for validation (default 0.10)
            test_ratio: Proportion of data for testing (default 0.10)
            random_seed: Random seed for reproducible splits
        """
        self.root = root
        full_data = pd.read_csv(csv_path)
        self.mode = mode
        self.augmentation = augmentation

        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "train_ratio + val_ratio + test_ratio must equal 1.0"
        assert mode in ["train", "val", "test"], \
            "mode must be one of 'train', 'val', or 'test'"

        # Create reproducible splits
        np.random.seed(random_seed)
        n_samples = len(full_data)
        indices = np.random.permutation(n_samples)

        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)

        if mode == "train":
            split_indices = indices[:train_end]
        elif mode == "val":
            split_indices = indices[train_end:val_end]
        else:  # test
            split_indices = indices[val_end:]

        self.data = full_data.iloc[split_indices].reset_index(drop=True)
        self.VAE = True

        self.cts = os.path.join(self.root , 'CT_preprocessed')
        self.xrays = os.path.join(self.root, 'PX_preprocessed')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ct_accession = str(self.data.loc[idx, CT_ACCESSION_COL]).strip()
        cxr_accession = str(self.data.loc[idx, XRAY_ACCESSION_COL]).strip()

        # Load CT (latent or volume)
        ct = np.load(os.path.join(self.cts, ct_accession + '.npy')).astype(np.float32)

        if not self.VAE:
            if self.augmentation:
                random_n = torch.rand(1)
                if random_n[0] > 0.5:
                    ct = np.flip(ct, 0)

        ctout = torch.from_numpy(ct.copy()).float()
        if not self.VAE:
            ctout = ctout.unsqueeze(0) #(num_slices,H,W) → (1,num_slices,H,W)
        else:
            ctout = ctout.permute(0,3,1,2) #(4,H/8,W/8,num_slices)-> (4,num_slices,H/8,W/8)

        # Load matching Xray 2D image
        xray = torch.from_numpy(np.load(os.path.join(self.xrays, cxr_accession + '.npy'))).float()

        if self.mode in ("train", "val", "test"):
            return {'ct': ctout, 'cxr': xray}
        else:  # infer mode
            return {'ct': ctout, 'cxr': xray, 'ct_accession': ct_accession, 'cxr_accession': cxr_accession}
