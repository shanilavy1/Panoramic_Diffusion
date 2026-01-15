import os
import shutil
import pydicom

def filter_facial(main_dir, output_dir):
    """
    For each patient folder:
    - If SeriesNumber=8 exists → save ONLY Series 8
    - Else if SeriesNumber=4 exists → save ONLY Series 4
    """

    os.makedirs(output_dir, exist_ok=True)

    for subdir in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir)

        if not os.path.isdir(subdir_path):
            continue

        series_dict = {}

        # Collect DICOM files
        dicom_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(".dcm")]
        if not dicom_files:
            continue

        for file in dicom_files:
            file_path = os.path.join(subdir_path, file)
            dcm = pydicom.dcmread(file_path, force=True)
            series_number = int(dcm[0x0020, 0x0011].value)  # Se tag

            series_dict.setdefault(series_number, []).append(file_path)

        if 8 in series_dict:
            selected_series = 8
        elif 4 in series_dict:
            selected_series = 4

        # Prepare output directory for this patient
        patient_out_dir = os.path.join(output_dir, subdir)
        os.makedirs(patient_out_dir, exist_ok=True)

        # Copy selected series files
        for file_path in series_dict[selected_series]:
            shutil.copy(file_path, patient_out_dir)
        print(f"Saved Series {selected_series} for patient {subdir}")


filter_facial(
    main_dir="/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/facial_bones_CT/facial_bones_2025_2020_spine",
    output_dir="/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/facial_bones_CT/facial_bones_2025_2020_spine_se8")

