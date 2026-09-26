from pathlib import Path
import pandas as pd 

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/"data"/"raw"/"Dry_Bean_Dataset"/"Dry_Bean_Dataset.xlsx"

def load_data(path=DATA_PATH):
    return pd.read_excel(path)
