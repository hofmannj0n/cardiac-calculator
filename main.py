import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# Load the model and the imputer
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('imputer.pkl', 'rb') as imputer_file:
    imputer = pickle.load(imputer_file)

# Define input variables
INPUT_VARS = ['age', 'bmi', 'andur', 'asa', 'emop',
              'preop_hb', 'preop_platelet', 'preop_wbc', 'preop_aptt', 'preop_ptinr', 'preop_glucose',
              'preop_bun', 'preop_ast', 'preop_alt', 'preop_creatinine', 'preop_sodium', 'preop_potassium',
              'Index', 'Elixhauser Indicator']

# Define a function to make predictions
def predict_post_op_complications(user_input):
    # Create a DataFrame from the user input
    input_df = pd.DataFrame([user_input], columns=INPUT_VARS)
    # Impute missing values
    input_imputed = imputer.transform(input_df)
    # Make prediction
    prediction = model.predict_proba(input_imputed)[:, 1][0]
    return prediction

# Streamlit UI
st.set_page_config(page_title='Predicting Post-Operative Death in Cardiac Patients', page_icon='🩺', layout='centered')
st.markdown('<h1 style="color: #4CAF50;">Predicting Post-Operative Death in Cardiac Patients</h1>', unsafe_allow_html=True)

st.markdown('''
<style>
    .main {
        background-color: #f5f5f5;
        color: #333333;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .stNumberInput > label {
        color: #333333;
    }
    .stAlert {
        background-color: #4CAF50;
        color: white;
    }
</style>
''', unsafe_allow_html=True)

st.write('Enter the following information to predict the likelihood of post-operative complications:')

# User inputs
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=25.0)
    andur = st.number_input('Andur', min_value=0, max_value=10, value=5)
    asa = st.number_input('ASA', min_value=1, max_value=5, value=3)
    emop = st.number_input('EMOP', min_value=0, max_value=1, value=0)
    preop_hb = st.number_input('Preop HB', min_value=0.0, max_value=20.0, value=13.5)
    preop_platelet = st.number_input('Preop Platelet', min_value=0.0, max_value=500.0, value=250.0)

with col2:
    preop_wbc = st.number_input('Preop WBC', min_value=0.0, max_value=50.0, value=7.0)
    preop_aptt = st.number_input('Preop APTT', min_value=0.0, max_value=100.0, value=30.0)
    preop_ptinr = st.number_input('Preop PTINR', min_value=0.0, max_value=10.0, value=1.0)
    preop_glucose = st.number_input('Preop Glucose', min_value=0.0, max_value=500.0, value=100.0)
    preop_bun = st.number_input('Preop BUN', min_value=0.0, max_value=100.0, value=15.0)
    preop_ast = st.number_input('Preop AST', min_value=0.0, max_value=100.0, value=20.0)

with col3:
    preop_alt = st.number_input('Preop ALT', min_value=0.0, max_value=100.0, value=20.0)
    preop_creatinine = st.number_input('Preop Creatinine', min_value=0.0, max_value=10.0, value=1.0)
    preop_sodium = st.number_input('Preop Sodium', min_value=0.0, max_value=200.0, value=140.0)
    preop_potassium = st.number_input('Preop Potassium', min_value=0.0, max_value=10.0, value=4.0)
    Index = st.number_input('Index', min_value=0, max_value=100, value=50)
    elixhauser_indicator = st.number_input('Elixhauser Indicator', min_value=0, max_value=1, value=0)

# Collect user input into a dictionary
user_input = {
    'age': age,
    'bmi': bmi,
    'andur': andur,
    'asa': asa,
    'emop': emop,
    'preop_hb': preop_hb,
    'preop_platelet': preop_platelet,
    'preop_wbc': preop_wbc,
    'preop_aptt': preop_aptt,
    'preop_ptinr': preop_ptinr,
    'preop_glucose': preop_glucose,
    'preop_bun': preop_bun,
    'preop_ast': preop_ast,
    'preop_alt': preop_alt,
    'preop_creatinine': preop_creatinine,
    'preop_sodium': preop_sodium,
    'preop_potassium': preop_potassium,
    'Index': Index,
    'Elixhauser Indicator': elixhauser_indicator
}

# When the user clicks the Predict button
if st.button('Predict'):
    # Make the prediction
    prediction = predict_post_op_complications(user_input)
    st.markdown(f'<div class="stAlert">According to LR model, the predicted likelihood of 30 day in hospital death is {prediction*100:.2f}%</div>', unsafe_allow_html=True)
