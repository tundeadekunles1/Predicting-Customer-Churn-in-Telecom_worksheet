# src/data_preprocessing.py

import os
import pandas as pd

def create_data_dirs():
    """Create raw and processed data directories if they don't exist."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, handle missing values."""
    df.drop_duplicates(inplace=True)
    df.dropna(how="any", inplace=True)
    return df
     
