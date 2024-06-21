# Post-Operative Complications Predictor 🩺

This repository contains a Streamlit web application for predicting the likelihood of post-operative complications. The application takes various pre-operative factors as inputs and uses a pre-trained logistic regression model to make predictions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Post-Operative Complications Predictor is designed to help healthcare professionals assess the risk of complications after surgery. By inputting patient data such as age, BMI, and various pre-operative test results, the app predicts the probability of post-operative complications.

## Features

- **User-Friendly Interface**: Enter patient data through an easy-to-use web interface.
- **Real-Time Predictions**: Get instant predictions based on the input data.
- **Detailed Inputs**: Includes a wide range of pre-operative factors for comprehensive risk assessment.
- **Visualization**: Clear and concise display of prediction results.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/post-op-complications-predictor.git
    cd post-op-complications-predictor
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

## Usage

1. Open your web browser and navigate to the local Streamlit server (usually `http://localhost:8501`).
2. Enter the required patient information into the input fields.
3. Click the **Predict** button to get the likelihood of post-operative complications.
4. The prediction result will be displayed on the screen.

## Model Training

The logistic regression model used in this app was trained using a dataset containing various pre-operative factors. Here is a brief overview of the training process:

1. **Data Preprocessing**: The data was shuffled, split into training and test sets, and balanced using SMOTE and undersampling.
2. **Model Training**: A logistic regression model was trained using the balanced dataset.
3. **Model Saving**: The trained model and imputer were saved using `pickle` for later use in the Streamlit app.

To train and save the model, use the following script:

```python
import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('your_data.csv')

# Define input and output variables
INPUT_VARS = ['age', 'bmi', 'andur', 'asa', 'emop', 'preop_hb', 'preop_platelet', 'preop_wbc',
              'preop_aptt', 'preop_ptinr', 'preop_glucose', 'preop_bun', 'preop_ast', 
              'preop_alt', 'preop_creatinine', 'preop_sodium', 'preop_potassium', 'Index', 
              'Elixhauser Indicator']
OUTCOME_VAR = 'outcome'

# Shuffle and split the dataset
df = df.sample(frac=1, random_state=1).reset_index(drop=True)
x_train, x_test, y_train, y_test = train_test_split(
    df[INPUT_VARS], df[OUTCOME_VAR], test_size=0.3, stratify=df[OUTCOME_VAR], random_state=1
)

# Apply SMOTE and undersampling
smote = SMOTE(random_state=1)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
rus = RandomUnderSampler(random_state=1)
x_train_balanced, y_train_balanced = rus.fit_resample(x_train_smote, y_train_smote)

# Impute missing values and train the model
imputer = SimpleImputer().fit(x_train_balanced)
x_train_imputed = imputer.transform(x_train_balanced)
model = LogisticRegression(max_iter=5000).fit(x_train_imputed, y_train_balanced)

# Save the model and imputer
with open('logistic_regression_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('imputer.pkl', 'wb') as imputer_file:
    pickle.dump(imputer, imputer_file)
