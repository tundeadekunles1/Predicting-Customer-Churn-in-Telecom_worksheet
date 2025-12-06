from data_preprocessing import full_preprocess_pipeline

RAW_PATH = "data/raw/Telco-Customer-Churn.csv"
PROCESSED_PATH = "data/processed/telco_processed.csv"

df_proc = full_preprocess_pipeline(RAW_PATH, PROCESSED_PATH)
print(df_proc.head())
