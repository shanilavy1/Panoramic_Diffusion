import os
import torch
import torch.utils.checkpoint
from params import *
import numpy as np
import scipy.ndimage
import warnings
import scipy
from scipy.ndimage import zoom
import pydicom as dicom
from diffusers import AutoencoderKL

# Statistics for Hounsfield Units
CONTRAST_HU_MIN = 300.     # Min value for loading contrast, bone window
CONTRAST_HU_MAX = 2000.      # Max value for loading contrast, bone window


def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def get_first_of_dicom_field_as_int(x):
    if type(x) == dicom.multival.MultiValue: return int(x[0])
    else: return int(x)

def make_rgb(volume):
    """Tile a NumPy array to make sure it has 3 channels."""
    z, c, h, w = volume.shape

    tiling_shape = [1]*(len(volume.shape))
    tiling_shape[1] = 3
    np_vol = torch.tile(volume, tiling_shape)
    return np_vol

def encode_ctpa(ct, vae):

    device = "cuda"
    weight_dtype =  torch.float32
    ct = torch.from_numpy(ct).unsqueeze(0).unsqueeze(0)

    print("ct before vae = ", ct.shape)
    vae.eval()

    latents = []
    with torch.no_grad():
        for i in range(ct.shape[2]):
            slice = make_rgb(ct[:,:,i,:,:]) #(1,3,256,256)
            latent = vae.encode(slice.to(device, dtype=weight_dtype)).latent_dist.sample() #(1,4,32,32)
            latents.append(latent)
        latents = torch.stack(latents, dim=4).squeeze()#.unsqueeze(0)#.permute(0,2,3,4,1) #(4,32,32, num_slices)

        latents = latents.detach().cpu().numpy()
        print("latents shape", latents.shape)
        latents = latents * 0.18215 #stable diffusion convention

    return latents

def normalize(img):
    
    img = img.astype(np.float32)
    img = (img - CONTRAST_HU_MIN) / (CONTRAST_HU_MAX - CONTRAST_HU_MIN) 
    img = np.clip(img, 0., 1.) * 2 -1
    return img

def resample_volume(volume, current_spacing, new_spacing):
    resize_factor = np.array(current_spacing) / new_spacing
    new_real_shape = volume.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / volume.shape
    new_spacing = np.array(current_spacing) / real_resize_factor

    print("new sampling = ", new_spacing)
    resampled_volume = scipy.ndimage.interpolation.zoom(volume, real_resize_factor)
    return resampled_volume

def resize_volume(tensor, output_size):

    z, h, w = tensor.shape
    resized_scan = np.zeros((output_size[0], output_size[1], output_size[2]))

    volume = tensor[:,:,:].squeeze()

    real_resize_factor = np.array(output_size) / np.shape(volume)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resized_scan[:,:,:] = scipy.ndimage.interpolation.zoom(volume, real_resize_factor, mode='nearest').astype(np.int16)

    return resized_scan


def preprocess_ctpa(img, attr, num_slices=150):
    """reshape, normalize image and convert to tensor"""

    # Slice-based cropping (instead of lung mask)
    z = img.shape[0]
    if z < num_slices:
        print(f"Warning: scan has only {z} slices, using all slices")
        img_cropped = img
    else:
        img_cropped = img[:num_slices, :, :]
    print("Cropping scan to first slices:", img_cropped.shape)

    # Resample
    spacing = attr['Spacing']
    img_resampled = resample_volume(img_cropped, spacing, [1,1,1])
    # Rescale
    img_resize = resize_volume(img_resampled, [128, 256, 256])
    # noramlize and window Hounsfield Units 
    img_normalized = normalize(img_resize)

    return img_normalized

def calculate_current_spacing(slices):
    spacings = []

    for slice in slices:
        pixel_spacing = slice.PixelSpacing
        slice_thickness = slice.SliceThickness

        # Convert the spacing values to floats
        pixel_spacing = [float(spacing) for spacing in pixel_spacing]
        slice_thickness = float(slice_thickness)

        spacings.append((pixel_spacing[0], pixel_spacing[1], slice_thickness)) #dont used

    spacing_CHECK = [float(slices[0].SliceThickness), 
                        float(slices[0].PixelSpacing[0]), 
                        float(slices[0].PixelSpacing[0])]

    return spacing_CHECK

def dicom_load_scan(paths):
    attr = {}
    slices = [dicom.dcmread(path) for path in paths]
    slices.sort(key = lambda x: int(x.InstanceNumber), reverse = True)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness

    spacing = calculate_current_spacing(slices)

    x, y = slices[0].PixelSpacing

    if slice_thickness == 0: #handle broken DICOMs
        attr['Spacing'] = spacing[0]
    else:
        attr['Spacing'] = (slice_thickness, x, y)

    attr['Position'] = slices[0].ImagePositionPatient
    attr['Orientation'] = slices[0].ImageOrientationPatient

    window_center, window_width, intercept, slope = get_windowing(slices[0])
    attr['window_center'] = window_center
    attr['window_width'] = window_width
    attr['intercept'] = intercept
    attr['slope'] = slope
    
    return (slices, attr)

def dicom_get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def preprocess_ctpa_directory(src_path, dst_path):

    device = DEVICE
    weight_dtype = torch.float32

    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    vae = AutoencoderKL.from_single_file(url)
    vae.to(device, dtype=weight_dtype)

    for accession_number in os.listdir(src_path):

        scan = os.path.join(src_path, accession_number)
        scan_dicom_files = [os.path.join(scan, f) for f in os.listdir(scan) if f.lower().endswith(".dcm")]
        scan_dst = os.path.join(dst_path, accession_number)

        print("processing scan - ", scan)

        slices, attr = dicom_load_scan(scan_dicom_files)
        ct = dicom_get_pixels_hu(slices) #convert pixels values to HU
        print("after loading exam = ", ct.shape)
        ct = preprocess_ctpa(ct, attr)
        print("after preprocessing = ", ct.shape)
        latents = encode_ctpa(ct, vae)
        print("scan_dst = ", scan_dst)

        np.save(scan_dst , latents)

    return

if __name__ == "__main__":
    src_path = "/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/CT_PX_dataset_diffusion/facial_bones_2025_2020_spine"
    dst_path = "/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/CT_PX_dataset_diffusion/CT_preprocessed"

    preprocess_ctpa_directory(src_path, dst_path)


