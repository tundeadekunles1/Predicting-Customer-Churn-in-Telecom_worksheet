import os
import joblib
import pandas as pd
import streamlit as st
# --------------------------------------------------
# Constants from your feature engineering
# --------------------------------------------------
MONTHLY_MEAN = 64.76169246059918  # from your notebook (np.float64)


# --------------------------------------------------
# Streamlit page config
# --------------------------------------------------
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    layout="centered"
)

st.title("Telecom Customer Churn Prediction")
st.write(
    "Enter customer information on the left to estimate the probability of churn "
    "using the trained Logistic Regression model from your DSML project."
)


# --------------------------------------------------
# Load trained pipeline (preprocessor + LogisticRegression)
# --------------------------------------------------
@st.cache_resource
def load_pipeline():
    model_path = os.path.join("models", "churn_pipeline.joblib")
    if not os.path.exists(model_path):
        st.error(
            "Model file `models/churn_pipeline.joblib` not found.\n\n"
            "Please run the notebook cell that saves the logistic regression pipeline "
            "as `churn_pipeline.joblib` inside the `models` folder."
        )
        st.stop()
    return joblib.load(model_path)


pipeline = load_pipeline()


# --------------------------------------------------
# Helper functions to build the 26 features
# --------------------------------------------------
def build_feature_row(
    senior_citizen: int,
    tenure: int,
    monthly_charges: float,
    total_charges: float,
    gender: str,
    partner: str,
    dependents: str,
    phone_service: str,
    multiple_lines: str,
    internet_service: str,
    online_security: str,
    online_backup: str,
    device_protection: str,
    tech_support: str,
    streaming_tv: str,
    streaming_movies: str,
    contract: str,
    paperless_billing: str,
    payment_method: str,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame with EXACTLY the 26 training features.
    """

    # --- Core numeric features ---
    charges_ratio = total_charges / tenure if tenure > 0 else 0.0

    high_spender = 1 if monthly_charges > MONTHLY_MEAN else 0

    high_churn_risk = 1 if (
        senior_citizen == 1
        and monthly_charges > MONTHLY_MEAN
        and tenure <= 12
    ) else 0

    # --- One-hot style binary flags from categorical inputs ---
    gender_male = 1 if gender == "Male" else 0

    partner_yes = 1 if partner == "Yes" else 0
    dependents_yes = 1 if dependents == "Yes" else 0

    phone_service_yes = 1 if phone_service == "Yes" else 0
    multiple_lines_yes = 1 if multiple_lines == "Yes" else 0

    internet_fiber = 1 if internet_service == "Fiber optic" else 0
    internet_no = 1 if internet_service == "No" else 0

    online_security_yes = 1 if online_security == "Yes" else 0
    online_backup_yes = 1 if online_backup == "Yes" else 0
    device_protection_yes = 1 if device_protection == "Yes" else 0
    tech_support_yes = 1 if tech_support == "Yes" else 0

    streaming_tv_yes = 1 if streaming_tv == "Yes" else 0
    streaming_movies_yes = 1 if streaming_movies == "Yes" else 0

    contract_one_year = 1 if contract == "One year" else 0
    contract_two_year = 1 if contract == "Two year" else 0

    paperless_billing_yes = 1 if paperless_billing == "Yes" else 0

    payment_credit_card = 1 if payment_method == "Credit card (automatic)" else 0
    payment_electronic_check = 1 if payment_method == "Electronic check" else 0
    payment_mailed_check = 1 if payment_method == "Mailed check" else 0

    # --- Build DataFrame with the 26 columns in your training list ---
    data = {
        "SeniorCitizen": [senior_citizen],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "charges_ratio": [charges_ratio],

        "gender_Male": [gender_male],
        "Partner_Yes": [partner_yes],
        "Dependents_Yes": [dependents_yes],
        "PhoneService_Yes": [phone_service_yes],
        "MultipleLines_Yes": [multiple_lines_yes],
        "InternetService_Fiber optic": [internet_fiber],
        "InternetService_No": [internet_no],
        "OnlineSecurity_Yes": [online_security_yes],
        "OnlineBackup_Yes": [online_backup_yes],
        "DeviceProtection_Yes": [device_protection_yes],
        "TechSupport_Yes": [tech_support_yes],
        "StreamingTV_Yes": [streaming_tv_yes],
        "StreamingMovies_Yes": [streaming_movies_yes],
        "Contract_One year": [contract_one_year],
        "Contract_Two year": [contract_two_year],
        "PaperlessBilling_Yes": [paperless_billing_yes],
        "PaymentMethod_Credit card (automatic)": [payment_credit_card],
        "PaymentMethod_Electronic check": [payment_electronic_check],
        "PaymentMethod_Mailed check": [payment_mailed_check],

        "HighSpender": [high_spender],
        "HighChurnRisk": [high_churn_risk],
    }

    return pd.DataFrame(data)


# --------------------------------------------------
# Sidebar: user inputs
# --------------------------------------------------
st.sidebar.header("Customer information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
)

tenure = st.sidebar.slider("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=250.0,
    value=70.0,
    step=1.0,
)
total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0,
    step=10.0,
)


# --------------------------------------------------
# Build input DataFrame and predict
# --------------------------------------------------
input_df = build_feature_row(
    senior_citizen=senior_citizen,
    tenure=tenure,
    monthly_charges=monthly_charges,
    total_charges=total_charges,
    gender=gender,
    partner=partner,
    dependents=dependents,
    phone_service=phone_service,
    multiple_lines=multiple_lines,
    internet_service=internet_service,
    online_security=online_security,
    online_backup=online_backup,
    device_protection=device_protection,
    tech_support=tech_support,
    streaming_tv=streaming_tv,
    streaming_movies=streaming_movies,
    contract=contract,
    paperless_billing=paperless_billing,
    payment_method=payment_method,
)

st.subheader("Model input features")
st.dataframe(input_df)


if st.button("Predict churn"):
    try:
        # Logistic Regression pipeline: preprocessor + classifier
        prob = pipeline.predict_proba(input_df)[:, 1][0]
        pred = pipeline.predict(input_df)[0]

        st.subheader("Prediction result")
        st.write(f"Estimated churn probability: **{prob:.2f}**")

        if int(pred) == 1:
            st.error("This customer is **likely** to churn.")
        else:
            st.success("This customer is **not likely** to churn.")

    except Exception as e:
        st.error(
            "An error occurred while making the prediction. "
            "Please ensure the saved `churn_pipeline.joblib` was trained "
            "on the same 26 features as constructed here."
        )
        st.exception(e)
