import pandas as pd
import numpy as np
from pathlib import Path

def create_data_dirs(base_path="../data"):
    """
    Create raw and processed data directories if they don't exist.
    """
    Path(f"{base_path}/raw").mkdir(parents=True, exist_ok=True)
    Path(f"{base_path}/processed").mkdir(parents=True, exist_ok=True)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco Customer Churn dataset:
    - Convert TotalCharges to numeric
    - Fill missing values with median
    - Drop duplicates
    
    Parameters:
        df (pd.DataFrame): Raw dataset.
    
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing values with median
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Drop duplicates if any
    if df.duplicated().any():
        print(f"Dropping {df.duplicated().sum()} duplicate rows...")
        df.drop_duplicates(inplace=True)
    else:
        print("No duplicate rows found.")
    
    return df

