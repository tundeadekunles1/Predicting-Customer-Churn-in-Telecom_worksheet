from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def prepare_data(df):
    """Encode, split, and scale dataset."""
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df.drop(columns=["customerID", "tenure_group"], axis=1, inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include="object"):
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression and return the fitted model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest and return the fitted model."""
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    return model
