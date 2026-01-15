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
    def __init__(self, root='.', csv_path=None, mode="train", augmentation=False):
        self.root = root
        self.data = pd.read_csv(csv_path)
        self.mode = mode
        self.augmentation = augmentation
        self.VAE = True

        self.cts = os.path.join(self.root , 'CT_preprocessed')
        self.xrays = os.path.join(self.root, 'PX_preprocessed')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ct_accession = self.data.loc[idx, CT_ACCESSION_COL]
        cxr_accession = self.data.loc[idx, XRAY_ACCESSION_COL]

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

        if self.mode == "train" or self.mode == "test":
            return {'ct': ctout, 'cxr': xray}
        else: #if self.mode == "infer"
            return {'ct': ctout, 'cxr': xray, 'ct_accession': ct_accession, 'cxr_accession': cxr_accession}
