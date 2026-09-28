import zipfile
from pathlib import Path
import numpy as np
import pandas as pd


def data_wrangler(dataset):
    dataset = dataset.copy()

    # Consolidated mapping configurations
    mappings = {
        (
            "BsmtQual",
            "ExterQual",
            "ExterCond",
            "BsmtCond",
            "HeatingQC",
            "KitchenQual",
            "FireplaceQu",
            "GarageQual",
            "GarageCond",
            "PoolQC",
        ): {
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "NA": 0,
            np.nan: 0,
        },
        ("BsmtExposure",): {"Gd": 4, "Av": 3, "Mn": 2, "No": 0.5, "NA": 0, np.nan: 0},
        ("BsmtFinType1", "BsmtFinType2"): {
            "GLQ": 6,
            "ALQ": 5,
            "BLQ": 4,
            "Rec": 3,
            "LwQ": 2,
            "Unf": 1,
            "NA": 0,
            np.nan: 0,
        },
        ("CentralAir",): {"Y": 1, "N": 0},
        ("Functional",): {
            "Typ": 8,
            "Min1": 7,
            "Min2": 6,
            "Mod": 5,
            "Maj1": 4,
            "Maj2": 3,
            "Sev": 2,
            "Sal": 1,
        },
        ("GarageType",): {
            "2Types": 6,
            "Attchd": 5,
            "Basment": 4,
            "BuiltIn": 3,
            "CarPort": 2,
            "Detchd": 1,
            "NA": 0,
            np.nan: 0,
        },
        ("GarageFinish",): {"Fin": 3, "RFn": 2, "Unf": 1, "NA": 0, np.nan: 0},
        ("PavedDrive",): {"Y": 2, "P": 1, "N": 0},
        ("Street",): {"Grvl": 1, "Pave": 2},
        ("LandSlope",): {"Gtl": 3, "Mod": 2, "Sev": 1},
        ("Heating",): {
            "Floor": 2,
            "GasA": 6,
            "GasW": 5,
            "Grav": 4,
            "OthW": 1,
            "Wall": 3,
        },
    }

    # Apply mappings
    for cols, mapper in mappings.items():
        for col in cols:
            if col in dataset.columns:
                dataset[col] = dataset[col].map(mapper)

    # DROP ID COLUMN HERE
    dataset = dataset.drop(columns=["Id"], errors="ignore")

    # Fill remaining NULL values
    num_cols = dataset.select_dtypes(include=["number"]).columns
    cat_cols = dataset.select_dtypes(exclude=["number"]).columns

    dataset[num_cols] = dataset[num_cols].fillna(0)
    dataset[cat_cols] = dataset[cat_cols].fillna("Not Available")

    return dataset


def unzip_files():
    data_path = Path.cwd() / "Data"
    file_path = data_path / "house-prices-advanced-regression-techniques.zip"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    dfs = {}
    with zipfile.ZipFile(file_path, "r") as zip_file:
        for filename in zip_file.namelist():
            if filename.endswith(".csv"):
                name = Path(filename).stem
                dfs[name] = pd.read_csv(zip_file.open(filename))

    return dfs.get("test"), dfs.get("sample_submission"), dfs.get("train")


import pandas as pd

def main_wrangler():
    file_path = r"C:\Users\Jelil\House_Sales\Data_folder\train_data.csv"
    
    try:
        dataset = pd.read_csv(file_path)
    except Exception as e:
        print(f"File not compatible or could not be loaded: {e}")
        return None

    if isinstance(dataset, pd.DataFrame):
        print("dataset is a DataFrame. Proceeding with analysis...")
        print(f"Dataset loaded has shape : {dataset.shape}")
        print(dataset.info())
        return dataset
    else:
        print("File not compatible")
        return None


if __name__ == "__main__":
    main_wrangler()
    