import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# 1. Page Configuration & Header
# =========================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Customer Churn Prediction App")
st.write("This application predicts whether a customer is likely to leave (Churn) using a trained Machine Learning model.")
st.divider()

# =========================================================
# 2. Load the Pre-trained Model and Scaler
# =========================================================
@st.cache_resource
def load_model():
    """
    Loads the trained model package saved during training.
    Contains: 'model' (RandomForest), 'scaler' (StandardScaler), 'feature_names'
    """
    model_path = "models/churn_model_package.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    return joblib.load(model_path)

model_package = load_model()

# =========================================================
# 3. User Input Interface (Customer Details)
# =========================================================
st.subheader("📋 Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract Type",
        options=["Month-to-month", "One year", "Two year"],
        help="Type of contract the customer currently holds."
    )
    
    tenure = st.slider(
        "Tenure (Months with company)",
        min_value=1,
        max_value=72,
        value=12,
        help="Number of months the customer has stayed with the company."
    )
    
    payment_method = st.selectbox(
        "Payment Method",
        options=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=10.0,
        max_value=150.0,
        value=65.0,
        step=1.0
    )
    
    # Auto-calculate reasonable total charges estimate based on tenure
    default_total = float(round(tenure * monthly_charges, 2))
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=10.0,
        max_value=10000.0,
        value=default_total,
        step=10.0
    )
    
    phone_service = st.radio("Phone Service", options=["Yes", "No"], horizontal=True)
    paperless_billing = st.radio("Paperless Billing", options=["Yes", "No"], horizontal=True)

st.divider()

# =========================================================
# 4. Feature Preprocessing (Encoding to match Model)
# =========================================================
def preprocess_input():
    # 1. Map Contract (Ordinal)
    contract_map = {"Month-to-month": 0.0, "One year": 1.0, "Two year": 2.0}
    ord_contract = contract_map[contract]

    # 2. Map Tenure Group
    if tenure <= 6:
        ord_tenure_group = 0.0  # 0-6 months
    elif tenure <= 24:
        ord_tenure_group = 1.0  # 7-24 months
    else:
        ord_tenure_group = 2.0  # 25+ months

    # 3. Binary & One-Hot Encoded features
    phone_yes = 1.0 if phone_service == "Yes" else 0.0
    paperless_yes = 1.0 if paperless_billing == "Yes" else 0.0
    pay_credit_card = 1.0 if payment_method == "Credit card (automatic)" else 0.0
    pay_elec_check = 1.0 if payment_method == "Electronic check" else 0.0
    pay_mailed_check = 1.0 if payment_method == "Mailed check" else 0.0

    # 4. Assemble the exact 10 features in required order
    feature_dict = {
        "ord__Contract": ord_contract,
        "ord__Tenure_Group": ord_tenure_group,
        "nom__PhoneService_Yes": phone_yes,
        "nom__PaperlessBilling_Yes": paperless_yes,
        "nom__PaymentMethod_Credit card (automatic)": pay_credit_card,
        "nom__PaymentMethod_Electronic check": pay_elec_check,
        "nom__PaymentMethod_Mailed check": pay_mailed_check,
        "remainder__tenure": float(tenure),
        "remainder__MonthlyCharges": float(monthly_charges),
        "remainder__TotalCharges": float(total_charges)
    }

    return pd.DataFrame([feature_dict])

# =========================================================
# 5. Prediction and Results Display
# =========================================================
if st.button("🔮 Predict Customer Churn", type="primary", use_container_width=True):
    if model_package is not None:
        model = model_package["model"]
        scaler = model_package["scaler"]
        
        # Prepare and scale the data
        input_df = preprocess_input()
        input_scaled = scaler.transform(input_df)

        # Get prediction and probabilities
        prediction = model.predict(input_scaled)[0]
        churn_probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("📊 Prediction Result")
        
        # Display probability progress bar
        st.write(f"**Churn Risk Score:** `{churn_probability * 100:.1f}%`")
        st.progress(float(churn_probability))

        # Show final verdict
        if prediction == 1:
            st.error(f"⚠️ **High Churn Risk!** This customer is likely to leave (Risk: {churn_probability*100:.1f}%).")
            st.info("💡 **Recommendation:** Consider offering a long-term contract discount or retention incentives.")
        else:
            st.success(f"✅ **Loyal Customer!** This customer is likely to stay (Risk: {churn_probability*100:.1f}%).")
            st.info("💡 **Recommendation:** Customer is satisfied. Maintain standard engagement.")

        # Optional: Show input data for doctor review
        with st.expander("🔍 View Processed Features (10 Model Inputs)"):
            st.dataframe(input_df)

# =========================================================
# 6. Sidebar (Project Info for Defense)
# =========================================================
with st.sidebar:
    st.header("ℹ️ Project Information")
    st.markdown("""
    - **Project:** Barrier - Churn Intelligence
    - **Model:** Random Forest Classifier
    - **Features:** 10 Preprocessed Inputs
    - **Scaler:** StandardScaler
    - **Framework:** Streamlit + Scikit-Learn
    """)
    st.divider()
    st.caption("NTI Graduation Project // Barrier-")
