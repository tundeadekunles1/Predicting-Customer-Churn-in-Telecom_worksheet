import streamlit as st
import pandas as pd
import os
from src.data_preprocessing import create_data_dirs, load_data, clean_data
from src.feature_engineering import engineer_features
from src.model_training import prepare_data, train_logistic_regression, train_random_forest
from src.evaluation import evaluate_model

# Configure page
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telecom Customer Churn Prediction")
st.markdown("An end-to-end DSML project to predict customer churn using Machine Learning")

# Sidebar with options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Home", "Model Training", "Make Prediction"])

if page == "Home":
    st.header("Welcome!")
    st.markdown("""
    This app predicts which telecom customers are likely to churn using the **IBM Telco Customer Churn dataset**.
    
    ### Key Features:
    - 📈 Exploratory Data Analysis
    - 🤖 Logistic Regression & Random Forest models
    - 🎯 Real-time churn predictions
    - 📊 Model performance metrics
    
    ### Business Impact:
    Even small increases in churn significantly impact recurring revenue. This model helps retention teams identify at-risk customers.
    """)

elif page == "Model Training":
    st.header("🤖 Model Training & Evaluation")
    
    if st.button("Train Models"):
        with st.spinner("Training models... This may take a minute."):
            try:
                # Create directories and load data
                create_data_dirs()
                df = load_data("data/raw/Telco-Customer-Churn.csv")
                
                # Preprocess and engineer features
                df = clean_data(df)
                df = engineer_features(df)
                
                # Prepare data
                X_train, X_test, y_train, y_test = prepare_data(df)
                
                # Train models
                log_model = train_logistic_regression(X_train, y_train)
                rf_model = train_random_forest(X_train, y_train)
                
                # Evaluate
                log_results = evaluate_model(log_model, X_test, y_test)
                rf_results = evaluate_model(rf_model, X_test, y_test)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Logistic Regression")
                    st.metric("Accuracy", f"{log_results['accuracy']:.4f}")
                    st.metric("Precision", f"{log_results['precision']:.4f}")
                    st.metric("Recall", f"{log_results['recall']:.4f}")
                    st.metric("F1-Score", f"{log_results['f1']:.4f}")
                
                with col2:
                    st.subheader("Random Forest")
                    st.metric("Accuracy", f"{rf_results['accuracy']:.4f}")
                    st.metric("Precision", f"{rf_results['precision']:.4f}")
                    st.metric("Recall", f"{rf_results['recall']:.4f}")
                    st.metric("F1-Score", f"{rf_results['f1']:.4f}")
                
                st.success("✅ Models trained successfully!")
                
            except FileNotFoundError:
                st.error("❌ Data file not found. Please ensure 'data/raw/Telco-Customer-Churn.csv' exists.")
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")

elif page == "Make Prediction":
    st.header("🎯 Predict Customer Churn")
    st.markdown("Enter customer information below to predict churn probability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 0, 120, 50)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
    with col2:
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    
    if st.button("Predict Churn"):
        st.info("Prediction feature requires a trained model. Train models first in the 'Model Training' section.")

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.info("Built with ❤️ using Streamlit | DSML Group Project")

