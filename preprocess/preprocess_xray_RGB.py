from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import pydicom
import os

def get_transform(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def format_xray(img, mean, std):

    transform_xray = get_transform(mean, std)

    # Normalize
    img = transform_xray(img).float()

    #Resize
    img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)

    formated_img = img.squeeze().detach().cpu().numpy()
    return formated_img

def preprocess_xray(img, mean, std):

    xray = np.moveaxis(np.repeat(np.expand_dims(img, axis=0), 3, axis=0), 0, -1)

    print("Original:   ", xray.shape) #(1024, 2218, 3)

    # resize and normalize
    xray = format_xray(xray, mean, std)
    print("after formatting:   ", xray.shape) #(3, 224, 224)

    return xray


def preprocess_xray_directory(src_path, dst_path, mean, std):
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

        #turning the numpy into 3 RGB
        x_processed = preprocess_xray(img, mean, std)

        # save
        out_path = os.path.join(dst_path, subdir + '.npy')
        np.save(out_path, x_processed)

        print(f"Saved: {out_path}")

    print("Done.")


def compute_xray_mean_std(src_path):
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    for subdir in os.listdir(src_path):
        sub_path = os.path.join(src_path, subdir)
        if not os.path.isdir(sub_path):
            continue

        # find DICOM file
        dcm_files = [f for f in os.listdir(sub_path) if f.lower().endswith('.dcm')]
        if not dcm_files:
            continue

        dcm_path = os.path.join(sub_path, dcm_files[0])
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array.astype(np.float32)

        # normalize to 0–1 for fair statistics
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        pixel_sum += img.sum()
        pixel_sq_sum += (img ** 2).sum()
        pixel_count += img.size

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean**2)
    return mean, std

if __name__ == "__main__":
    src_path_px = "/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/CT_PX_dataset_diffusion/PX_2025_2020_spine"
    dst_path_px = "/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/CT_PX_dataset_diffusion/PX_preprocessed_RGB"
    mean_px, std_px = compute_xray_mean_std(src_path_px)
    print("Computed mean/std:", mean_px, std_px) #0.48354707042746453 0.24127288155377674
    preprocess_xray_directory(src_path_px, dst_path_px, mean_px, std_px)
