from dataset import XrayCTPADataset
from torch.utils.data import WeightedRandomSampler
from params import CSV_PATH

def get_dataset(cfg):

    if cfg.dataset.name == 'XRAY_CTPA':
        train_dataset = XrayCTPADataset(root=cfg.dataset.root_dir, csv_path=CSV_PATH, mode="train", augmentation=True)
        val_dataset = XrayCTPADataset(root=cfg.dataset.root_dir, csv_path=CSV_PATH, mode="val", augmentation=False)
        test_dataset = XrayCTPADataset(root=cfg.dataset.root_dir, csv_path=CSV_PATH, mode="test", augmentation=False)
        sampler = None
        return train_dataset, val_dataset, test_dataset, sampler

    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
