import torch.nn.functional as F
import numpy as np
import pydicom
import os
import torch


def preprocess_xray(img):
    """
    Normalize and resize a panoramic X-ray image.
    Uses per-image min-max normalization to [-1, 1] range,
    which is the standard input range for diffusion models.
    """
    img = img.astype(np.float32)

    # Per-image min-max normalization to [0, 1]
    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min + 1e-8)

    # Scale to [-1, 1] (standard diffusion model input range)
    img = img * 2.0 - 1.0

    print("Original:", img.shape)  # (H, W)=(1024, 2218)
    print(f"  After normalization: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}")

    # to tensor [1, H, W]
    img = torch.from_numpy(img).unsqueeze(0)
    # Resize [1, 1, 224, 224]
    img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    formated_img = img.squeeze(0).detach().cpu().numpy()  # remove batch dim only → [1, 224, 224]

    print("after formatting:", formated_img.shape)  # (1, 224, 224)

    return formated_img


def preprocess_xray_directory(src_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)

    for subdir in os.listdir(src_path):
        sub_path = os.path.join(src_path, subdir)
        if not os.path.isdir(sub_path):
            continue

        # find DICOM file in this folder
        dcm_files = [f for f in os.listdir(sub_path) if f.lower().endswith('.dcm')]
        if not dcm_files:
            continue

        dcm_path = os.path.join(sub_path, dcm_files[0])

        # load DICOM grayscale image
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array.astype(np.float32)

        # Handle MONOCHROME1 (inverted polarity: higher values = darker)
        # Convert to MONOCHROME2 convention (higher values = brighter)
        if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == 'MONOCHROME1':
            img = img.max() - img
            print(f"  Inverted MONOCHROME1 → MONOCHROME2 for {subdir}")

        # normalize and resize
        x_processed = preprocess_xray(img)

        # save
        out_path = os.path.join(dst_path, subdir + '.npy')
        np.save(out_path, x_processed)

        print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    src_path_px = "/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/CT_PX_dataset_diffusion/PX_2025_2020_spine"
    dst_path_px = "/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/CT_PX_dataset_diffusion/PX_preprocessed"
    preprocess_xray_directory(src_path_px, dst_path_px)
