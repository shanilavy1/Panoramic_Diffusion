from dataset import XrayCTPADataset
from params import CSV_PATH, TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH


def get_dataset(cfg):
    """Create train, validation, and test datasets.

    When use_augmented_data=True:
      - Train loads train_augmented.csv (original + augmented, train patients only)
      - Val loads val.csv (original only, val patients only)
      - Test loads test.csv (original only, test patients only)
      All from augmented directories. CSVs are pre-split — no data leakage.

    When use_augmented_data=False:
      - All three load the original CSV and split internally using ratios + seed.
      - From original directories (CT_preprocessed / PX_preprocessed).
    """
    if cfg.dataset.name == 'XRAY_CTPA':
        use_aug = cfg.model.use_augmented_data
        seed = cfg.model.seed

        if use_aug:
            # Pre-split CSVs from preprocess_augmented.py
            train_dataset = XrayCTPADataset(
                csv_path=TRAIN_CSV_PATH, root=cfg.dataset.root_dir,
                mode="train", use_augmented_data=True,
            )
            val_dataset = XrayCTPADataset(
                csv_path=VAL_CSV_PATH, root=cfg.dataset.root_dir,
                mode="val", use_augmented_data=True,
            )
            test_dataset = XrayCTPADataset(
                csv_path=TEST_CSV_PATH, root=cfg.dataset.root_dir,
                mode="test", use_augmented_data=True,
            )
        else:
            # Fallback: split original CSV internally
            train_dataset = XrayCTPADataset(
                csv_path=CSV_PATH, root=cfg.dataset.root_dir, mode="train",
                use_augmented_data=False, train_ratio=cfg.model.train_ratio,
                val_ratio=cfg.model.val_ratio, test_ratio=cfg.model.test_ratio,
                random_seed=seed,
            )
            val_dataset = XrayCTPADataset(
                csv_path=CSV_PATH, root=cfg.dataset.root_dir, mode="val",
                use_augmented_data=False, train_ratio=cfg.model.train_ratio,
                val_ratio=cfg.model.val_ratio, test_ratio=cfg.model.test_ratio,
                random_seed=seed,
            )
            test_dataset = XrayCTPADataset(
                csv_path=CSV_PATH, root=cfg.dataset.root_dir, mode="test",
                use_augmented_data=False, train_ratio=cfg.model.train_ratio,
                val_ratio=cfg.model.val_ratio, test_ratio=cfg.model.test_ratio,
                random_seed=seed,
            )

        return train_dataset, val_dataset, test_dataset

    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
