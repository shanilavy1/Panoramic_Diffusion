import pandas as pd

def excel_to_clean_csv(excel_path, csv_path):
    df = pd.read_excel(excel_path)

    # Columns to clean
    cols_to_clean = ["CT_ACCESSION_COL", "XRAY_ACCESSION_COL"]

    for col in cols_to_clean:
        df[col] = df[col].apply(clean_list_cell)

    df.to_csv(csv_path, index=False, encoding="utf-8")
    print("Saved cleaned CSV to:", csv_path)


def clean_list_cell(x):
    """
    Converts [12345] or '[12345]' → '12345'
    """
    if pd.isna(x):
        return x

    # If already numeric
    if isinstance(x, (int, float)):
        return int(x)

    # Convert string like "[12345]" or "[12345,]"
    x = str(x).strip()
    x = x.replace('[', '').replace(']', '').replace(',', '').strip()

    return x


if __name__ == "__main__":
    excel_file = "/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/CT_PX_dataset_diffusion/Facial_PX_dataset.xlsx"
    csv_file = "/media/rotem/7045cbc8-3ee4-485f-a6ea-18f7520d2704/Shani/Carotid_Plaque_data/CT_PX_dataset_diffusion/Facial_PX_dataset.csv"

    excel_to_clean_csv(excel_file, csv_file)
