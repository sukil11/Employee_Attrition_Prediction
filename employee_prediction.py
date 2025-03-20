import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load pre-trained models and scaler
rf_model = joblib.load('random_forest_model.joblib')
dt_model = joblib.load('decision_tree_model.joblib')
scaler = joblib.load('scaler.joblib')

# Set the background image
def set_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center; 
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
set_bg_image("https://taggd.in/wp-content/uploads/2023/06/Employee-Attrition-Banner-Image-2.png")

# Set up Streamlit interface
st.title("Employee Attrition Prediction")

# Input fields
st.header("Enter Employee Information")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
daily_rate = st.number_input("DailyRate", min_value=0, value=500)
distance_from_home = st.number_input("DistanceFromHome", min_value=0, max_value=100, value=10)
education = st.number_input("Education", min_value=1, max_value=4, value=2)
environment_satisfaction = st.number_input("EnvironmentSatisfaction", min_value=1, max_value=4, value=3)
hourly_rate = st.number_input("HourlyRate", min_value=0, value=20)
job_involvement = st.number_input("JobInvolvement", min_value=1, max_value=4, value=3)
job_level = st.number_input("JobLevel", min_value=1, max_value=5, value=2)
job_satisfaction = st.number_input("JobSatisfaction", min_value=1, max_value=4, value=3)
monthly_income = st.number_input("MonthlyIncome", min_value=1000, max_value=20000, value=5000)
num_companies_worked = st.number_input("NumCompaniesWorked", min_value=0, max_value=10, value=2)
over_time = st.selectbox("OverTime", options=[0, 1])  # 0 = No, 1 = Yes

# Create a DataFrame from the input
user_input = pd.DataFrame({
    'Age': [age],
    'DailyRate': [daily_rate],
    'DistanceFromHome': [distance_from_home],
    'Education': [education],
    'EnvironmentSatisfaction': [environment_satisfaction],
    'HourlyRate': [hourly_rate],
    'JobInvolvement': [job_involvement],
    'JobLevel': [job_level],
    'JobSatisfaction': [job_satisfaction],
    'MonthlyIncome': [monthly_income],
    'NumCompaniesWorked': [num_companies_worked],
    'OverTime': [over_time]
})

# Apply one-hot encoding for the categorical features like in the training set
user_input_encoded = pd.get_dummies(user_input, columns=['OverTime'], drop_first=True)

# Add missing columns based on the training set (to match column order)
trained_columns = joblib.load('trained_columns.joblib')  # Load the column names from the training phase
user_input_encoded = user_input_encoded.reindex(columns=trained_columns, fill_value=0)

# Scale the user input data using the same scaler
user_input_scaled = scaler.transform(user_input_encoded)

# Predictions
rf_pred = rf_model.predict(user_input_scaled)
dt_pred = dt_model.predict(user_input_scaled)

# Display predictions
st.subheader("Predictions")

if rf_pred == 1:
    st.write("Random Forest Model Prediction: Employee will leave.")
else:
    st.write("Random Forest Model Prediction: Employee will stay.")

if dt_pred == 1:
    st.write("Decision Tree Model Prediction: Employee will leave.")
else:
    st.write("Decision Tree Model Prediction: Employee will stay.")
