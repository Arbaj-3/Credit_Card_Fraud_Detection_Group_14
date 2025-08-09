import streamlit as st
import pandas as pd
import joblib


st.set_page_config(layout="wide")
st.sidebar.markdown(
    "Credit Card Fraud Detection App | <span style ='color: #7ac0c0;'>PG-DBDA Group-14</span>",
    unsafe_allow_html=True)

st.sidebar.write("[Tableau Visualizations](https://public.tableau.com/app/profile/arbaj.shaikh7124/viz/CreditCardFraudDetectionDashboard_17544252340210/CreditCardFraudDetectionDashboard?publish=yes)")

# Load the trained model
try:
    model = joblib.load('Best_Fraud_Model_XGBoost.pkl')
except FileNotFoundError:
    st.error("Error: The model file 'Best_Fraud_Model_XGBoost.pkl' was not found.")
    st.stop()

# Get the feature names from your DataFrame
# This list must match the columns used for training the model
feature_names = ['amt', 'age', 'distance', 'amt_per_capita', 'is_far',
                 'hour_bucket_encoded', 'category_grocery_pos', 'category_home',
                 'category_shopping_pos', 'category_kids_pets', 'category_shopping_net',
                 'category_entertainment', 'category_food_dining',
                 'category_personal_care', 'category_health_fitness',
                 'category_misc_pos', 'category_misc_net', 'category_grocery_net',
                 'category_travel', 'gender_M', 'city_freq']

st.title("Credit Card Fraud Detection")
st.write("Enter the transaction details to predict if it's fraudulent.")

# Create input widgets for each feature
input_data = {}

# Use columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Transaction Details")
    input_data['amt'] = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")
    input_data['age'] = st.slider("Age of Cardholder", 18, 100, 30)
    input_data['distance'] = st.number_input("Distance from Home to Transaction Location", min_value=0.0, format="%.2f")
    input_data['amt_per_capita'] = st.number_input("Amount per capita", min_value=0.0, format="%.2f")
    input_data['is_far'] = st.selectbox("Is Transaction Far from Home?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    input_data['gender_M'] = st.selectbox("Gender of Cardholder", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")

with col2:
    st.header("Transaction Time & Location")
    input_data['hour_bucket_encoded'] = st.slider("Time of Day (0-23)", 0, 23, 12)
    input_data['city_freq'] = st.number_input("City Frequency", min_value=0.0, format="%.2f")
    
    

with col3:
    # Category checkboxes
    st.header("Transaction Categories")
    st.markdown("_Select all that apply_")
    category_cols = ['category_grocery_pos', 'category_home', 'category_shopping_pos',
                     'category_kids_pets', 'category_shopping_net', 'category_entertainment',
                     'category_food_dining', 'category_personal_care', 'category_health_fitness',
                     'category_misc_pos', 'category_misc_net', 'category_grocery_net', 'category_travel']

    for col in category_cols:
        input_data[col] = st.checkbox(col.replace('category_', '').replace('_', ' ').title(), value=False)
    
    # Convert boolean checkboxes to integers
    for col in category_cols:
        input_data[col] = 1 if input_data[col] else 0

# Convert the input data to a DataFrame
input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    # Reorder columns to match the training data
    input_df = input_df[feature_names]

    # Make prediction
    prediction = model.predict(input_df)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
        st.write("Prediction: **Yes**")
    else:
        st.success("✅ Normal Transaction")
        st.write("Prediction: **No**")