# Model Deployment using All in One Method

import streamlit as st
import pickle
import pandas as pd
from tensorflow import keras
from keras.models import load_model

st.set_page_config(
    page_title="Milestone 1",
    page_icon= "🍂" ,
    layout="centered", 
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Customer Churn Prediction"
    }
)

# load scaler
f = open("preprocessor.pkl", "rb")
ss = pickle.load(f)
f.close()

# Model
model=load_model('gfgModel.h5')

columns=['SeniorCitizen', 'Partner', 'Dependents', 'PaperlessBilling', 'InternetService', 'OnlineSecurity', 
         'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
         'PaymentMethod', 'tenure', 'MonthlyCharges', 'TotalCharges']

# Title
st.markdown("<h1 style='text-align: center; color: black;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)

st.subheader("Basic Information:")
col1, col2 = st.columns(2)
with col1:
    SeniorCitizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
    Partner = st.selectbox("Partner", ['No', 'Yes'])
with col2:
    Dependents = st.selectbox("Dependents", ['No', 'Yes'])

st.subheader("Services Information:")
InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
col1, col2 = st.columns(2)
with col1:
    OnlineSecurity = st.selectbox("Online Security", ['No', 'No internet service', 'Yes'])
    OnlineBackup = st.selectbox("Online Backup", ['No', 'No internet service', 'Yes'])
    DeviceProtection = st.selectbox("Device Protection", ['No', 'No internet service', 'Yes'])
with col2:
    TechSupport = st.selectbox("Technical Support", ['No', 'No internet service', 'Yes'])
    StreamingTV = st.selectbox("Streaming TV", ['No', 'No internet service', 'Yes'])
    StreamingMovies = st.selectbox("Streaming Movies", ['No', 'No internet service', 'Yes'])

st.subheader("Customer Account Information:")
col1, col2 = st.columns(2)
with col1:
    PaperlessBilling = st.selectbox("Paperless Billing", ['No', 'Yes'])
    Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    PaymentMethod = st.selectbox("Payment Method", ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])
with col2:
    tenure = st.number_input("Tenure (months)")
    MonthlyCharges = st.number_input("Monthly Charges (USD)")
    TotalCharges = st.number_input("Total Charges (USD)")

# Data Inference
new_data = [SeniorCitizen, Partner, Dependents, PaperlessBilling, InternetService, OnlineSecurity, 
            OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, 
            PaymentMethod, tenure, MonthlyCharges, TotalCharges]
new_data = pd.DataFrame([new_data], columns=columns)
new_data_scaled = ss.transform(new_data)
pred = model.predict(new_data_scaled)
pred[pred <= 0.5] = 0
pred[pred > 0.5] = 1
pred = pred.squeeze() 

# Button to Predict
button = st.button("Predict")

if button:
    if pred == 1:
        st.write('Too bad! 😢')
        st.write('Your customer will **churn**. She/he will chooses to stop using your products/services.')
    else:
        st.write('Well done! 😄')
        st.write('Your customer will **stay (not churn)**. She/he will chooses to keep using your products/services.')