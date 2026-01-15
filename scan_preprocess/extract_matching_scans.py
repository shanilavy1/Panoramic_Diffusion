import pandas as pd
import os

def match_patient_ids(file1_path, file2_path, output1_path, output2_path, combined_output_path):

    # Read Excel files
    df1 = pd.read_excel(file1_path)
    df2 = pd.read_excel(file2_path)

    df1['AccessionNumber'] = df1['AccessionNumber'].apply(str)
    df1['PatientID'] = df1['PatientID'].apply(str)
    df2['AccessionNumber'] = df2['AccessionNumber'].apply(str)
    df2['PatientID'] = df2['PatientID'].apply(str)

    # Find matching PatientIDs
    matching_ids = set(df1['PatientID']).intersection(set(df2['PatientID']))

    if not matching_ids:
        print("No matching PatientIDs found!")
        return 0

    # Filter for matching records and keep only AccessionNumber
    matches1 = df1[df1['PatientID'].isin(matching_ids)][['AccessionNumber']]
    matches2 = df2[df2['PatientID'].isin(matching_ids)][['AccessionNumber']]

    # Save to Excel files
    matches1.to_excel(output1_path, index=False)
    matches2.to_excel(output2_path, index=False)

    print(f"Found {len(matching_ids)} matching PatientIDs")
    print(f"Found {len(matches1)} matching rows in file1 and {len(matches2)} in file2")
    print(f"File 1 matches saved to: {output1_path}")
    print(f"File 2 matches saved to: {output2_path}")

    # NEW: combined table
    result_rows = []

    for pid in matching_ids:
        acc1 = df1[df1['PatientID'] == pid]['AccessionNumber'].tolist()
        acc2 = df2[df2['PatientID'] == pid]['AccessionNumber'].tolist()

        result_rows.append({
            "PatientID": pid,
            "AccessionNumbers_File1": acc1,
            "AccessionNumbers_File2": acc2
        })

    result_df = pd.DataFrame(result_rows)
    result_df.to_excel(combined_output_path, index=False)

    print(f"Combined ID-to-list mapping saved to: {combined_output_path}")

    return len(matching_ids)


os.chdir('/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/tables/CT_PX_diffusion')
file1_path="Facial_2025_2015.xlsx"
file2_path="PX_2025_2020.xlsx"
output1_path="Facial_2025_2020_Accession.xlsx"
output2_path="PX_2025_2020_Accession.xlsx"
combined_output_path ="Facial_PX_2025_2020.xlsx"
match_patient_ids(file1_path, file2_path, output1_path, output2_path,combined_output_path)


