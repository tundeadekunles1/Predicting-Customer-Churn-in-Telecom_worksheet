from src.data_preprocessing import create_data_dirs, load_data, clean_data
from src.feature_engineering import engineer_features
from src.model_training import prepare_data, train_logistic_regression, train_random_forest

def main():
    create_data_dirs()
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    df = clean_data(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = prepare_data(df)

    print("Logistic Regression Results:")
    print(train_logistic_regression(X_train, y_train, X_test, y_test))

    print("\nRandom Forest Results:")
    print(train_random_forest(X_train, y_train, X_test, y_test))

if __name__ == "__main__":
    main()