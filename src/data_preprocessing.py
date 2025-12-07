import os
import pandas as pd

def create_data_dirs():
    """Create raw and processed data directories."""
    dirs = ["data/raw", "data/processed"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created or verified: {d}")

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV file into a DataFrame."""
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Telco churn dataset."""
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df.drop_duplicates(inplace=True)
    return df
