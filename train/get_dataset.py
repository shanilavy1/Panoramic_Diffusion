from dataset import XrayCTPADataset
from params import CSV_PATH


def get_dataset(cfg):
    """Create train, validation, and test datasets with configurable split ratios."""

    if cfg.dataset.name == 'XRAY_CTPA':
        train_ratio = cfg.model.train_ratio
        val_ratio = cfg.model.val_ratio
        test_ratio = cfg.model.test_ratio
        seed = cfg.model.seed

        train_dataset = XrayCTPADataset(
            root=cfg.dataset.root_dir, csv_path=CSV_PATH, mode="train",
            augmentation=True, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
            random_seed=seed
        )
        val_dataset = XrayCTPADataset(
            root=cfg.dataset.root_dir, csv_path=CSV_PATH, mode="val",
            augmentation=False, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
            random_seed=seed
        )
        test_dataset = XrayCTPADataset(
            root=cfg.dataset.root_dir, csv_path=CSV_PATH, mode="test",
            augmentation=False, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
            random_seed=seed
        )
        return train_dataset, val_dataset, test_dataset

    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
