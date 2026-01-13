import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title='Customer Churn Prediction App',
    page_icon='📱',
    layout='centered'
)

st.title('Customer Churn Prediction App')
st.write('Enter customer details to predict churn.')

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler

model = load_model()
scaler = load_scaler()

st.sidebar.header('Customer Details')

# Input fields for original features
account_length = st.sidebar.number_input('Account Length', min_value=1, max_value=250, value=100)
voice_mail_plan = st.sidebar.selectbox('Voice Mail Plan', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=0)
voice_mail_messages = st.sidebar.number_input('Voice Mail Messages', min_value=0, max_value=60, value=0)
day_mins = st.sidebar.number_input('Day Minutes', min_value=0.0, max_value=400.0, value=180.0, format="%.1f")
evening_mins = st.sidebar.number_input('Evening Minutes', min_value=0.0, max_value=400.0, value=200.0, format="%.1f")
night_mins = st.sidebar.number_input('Night Minutes', min_value=0.0, max_value=400.0, value=200.0, format="%.1f")
international_mins = st.sidebar.number_input('International Minutes', min_value=0.0, max_value=20.0, value=10.0, format="%.1f")
customer_service_calls = st.sidebar.number_input('Customer Service Calls', min_value=0, max_value=10, value=1)
international_plan = st.sidebar.selectbox('International Plan', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=0)
day_calls = st.sidebar.number_input('Day Calls', min_value=0, max_value=170, value=100)
day_charge = st.sidebar.number_input('Day Charge', min_value=0.0, max_value=70.0, value=30.0, format="%.2f")
evening_calls = st.sidebar.number_input('Evening Calls', min_value=0, max_value=170, value=100)
evening_charge = st.sidebar.number_input('Evening Charge', min_value=0.0, max_value=35.0, value=17.0, format="%.2f")
night_calls = st.sidebar.number_input('Night Calls', min_value=0, max_value=180, value=100)
night_charge = st.sidebar.number_input('Night Charge', min_value=0.0, max_value=20.0, value=9.0, format="%.2f")
international_calls = st.sidebar.number_input('International Calls', min_value=0, max_value=20, value=4)
international_charge = st.sidebar.number_input('International Charge', min_value=0.0, max_value=6.0, value=2.7, format="%.2f")
total_charge = st.sidebar.number_input('Total Charge', min_value=0.0, max_value=100.0, value=60.0, format="%.2f")

# Create a dictionary from the input values
input_dict = {
    'account_length': account_length,
    'voice_mail_plan': voice_mail_plan,
    'voice_mail_messages': voice_mail_messages,
    'day_mins': day_mins,
    'evening_mins': evening_mins,
    'night_mins': night_mins,
    'international_mins': international_mins,
    'customer_service_calls': customer_service_calls,
    'international_plan': international_plan,
    'day_calls': day_calls,
    'day_charge': day_charge,
    'evening_calls': evening_calls,
    'evening_charge': evening_charge,
    'night_calls': night_calls,
    'night_charge': night_charge,
    'international_calls': international_calls,
    'international_charge': international_charge,
    'total_charge': total_charge
}

# Convert the dictionary to a pandas DataFrame
original_features_df = pd.DataFrame([input_dict])

# Feature Engineering
original_features_df['total_calls'] = original_features_df['day_calls'] + original_features_df['evening_calls'] + original_features_df['night_calls'] + original_features_df['international_calls']
original_features_df['average_charge_per_call'] = original_features_df['total_charge'] / original_features_df['total_calls']
# Handle potential division by zero or inf values
original_features_df['average_charge_per_call'] = original_features_df['average_charge_per_call'].replace([np.inf, -np.inf], 0).fillna(0)
original_features_df['customer_lifetime_value'] = original_features_df['total_charge'] * 2
original_features_df['has_voicemail'] = original_features_df['voice_mail_plan'].apply(lambda x: 1 if x == 1 else 0)
original_features_df['has_international_plan'] = original_features_df['international_plan'].apply(lambda x: 1 if x == 1 else 0)

# Define the exact column order that the model was trained on
# This assumes the 'df' used in training had these columns in this order
# (Original 18 features + 5 engineered features, excluding 'churn')
expected_feature_order = [
    'account_length',
    'voice_mail_plan',
    'voice_mail_messages',
    'day_mins',
    'evening_mins',
    'night_mins',
    'international_mins',
    'customer_service_calls',
    'international_plan',
    'day_calls',
    'day_charge',
    'evening_calls',
    'evening_charge',
    'night_calls',
    'night_charge',
    'international_calls',
    'international_charge',
    'total_charge',
    'total_calls',
    'average_charge_per_call',
    'customer_lifetime_value',
    'has_voicemail',
    'has_international_plan'
]

# Create the final input DataFrame with the correct column order
input_df = original_features_df[expected_feature_order]

# Scale the input features
scaled_input = scaler.transform(input_df)

# Prediction button
if st.button('Predict Churn'):
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error('Customer is likely to churn.')
    else:
        st.success('Customer is not likely to churn.')
