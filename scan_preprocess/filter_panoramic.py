import os
import shutil
import pydicom

def filter_panoramic(main_dir, output_dir):
    """the function remove scan with RESEARCH ONLY in  the Series Description"""

    os.makedirs(output_dir, exist_ok=True)

    for subdir in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir)

        if not os.path.isdir(subdir_path):
            continue

        # Collect DICOM files
        dicom_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(".dcm")]
        if not dicom_files:
            continue

        for file in dicom_files:
            file_path = os.path.join(subdir_path, file)
            dcm = pydicom.dcmread(file_path, force=True)

            has_series_desc_tag = (0x0008, 0x103E) in dcm #Series Description tag

            print(subdir)
            print("Has SeriesDescription:", has_series_desc_tag)

            if has_series_desc_tag :
                print(f"Skipping (tag exists): {file_path}")
                continue

            out_subdir_path = os.path.join(output_dir, subdir)
            os.makedirs(out_subdir_path, exist_ok=True)
            out_file_path = os.path.join(out_subdir_path, file)
            shutil.copy2(file_path, out_file_path)

    print("Done.")



filter_panoramic(
    main_dir="/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/PX/PX_2025_2020_spine",
    output_dir="/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/PX/PX_2025_2020_spine_no_research",
)

