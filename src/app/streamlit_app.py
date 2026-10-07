import sys
from pathlib import Path

import streamlit as st


# Make the src package importable when Streamlit is launched from any directory.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from serving.inference import predict


st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="centered",
)

st.title("Telco Customer Churn Predictor")
st.write("Enter customer details to estimate churn risk.")

SAVED_EXAMPLES = {
    "Custom customer": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "tenure": 1,
        "MonthlyCharges": 85.0,
        "TotalCharges": 85.0,
    },
    "High-risk new customer": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "tenure": 1,
        "MonthlyCharges": 85.0,
        "TotalCharges": 85.0,
    },
    "Low-risk long-term customer": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "tenure": 60,
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
    },
    "Low-risk fiber customer": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "tenure": 48,
        "MonthlyCharges": 95.0,
        "TotalCharges": 4560.0,
    },
}

selected_example = st.selectbox("Load a saved customer example", list(SAVED_EXAMPLES))
example = SAVED_EXAMPLES[selected_example]

with st.expander("Model evaluation metrics"):
    st.write("Accuracy was not logged for the bundled MLflow run.")
    st.write("Precision: 0.490 | Recall: 0.821 | F1: 0.614 | ROC-AUC: 0.837")

with st.form("customer_form"):
    st.subheader("Customer details")

    gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(example["gender"]))
    senior_citizen = st.selectbox("Senior citizen", [0, 1], index=example["SeniorCitizen"], format_func=lambda value: "Yes" if value else "No")
    partner = st.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(example["Partner"]))
    dependents = st.selectbox("Dependents", ["Yes", "No"], index=["Yes", "No"].index(example["Dependents"]))

    phone_service = st.selectbox("Phone service", ["Yes", "No"], index=["Yes", "No"].index(example["PhoneService"]))
    multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(example["MultipleLines"]))
    internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(example["InternetService"]))
    online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(example["OnlineSecurity"]))
    online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(example["OnlineBackup"]))
    device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(example["DeviceProtection"]))
    tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(example["TechSupport"]))
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(example["StreamingTV"]))
    streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(example["StreamingMovies"]))

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(example["Contract"]))
    paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"], index=["Yes", "No"].index(example["PaperlessBilling"]))
    payment_method = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        index=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ].index(example["PaymentMethod"]),
    )

    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=example["tenure"], step=1)
    monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, value=example["MonthlyCharges"], step=0.01)
    total_charges = st.number_input("Total charges ($)", min_value=0.0, value=example["TotalCharges"], step=0.01)

    submitted = st.form_submit_button("Predict churn")

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        result = predict(customer)
        if result == "Likely to churn":
            st.warning(result)
        else:
            st.success(result)
    except Exception as error:
        st.error(f"Prediction failed: {error}")
