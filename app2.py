import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

model = pickle.load(open('model.pkl', 'rb'))

# Define the input fields for the app
def user_input_features():
    # Define the options and their meanings based on the DataFrame
    sex_meanings = {0: 'Female', 1: 'Male'}
    sex_options = sex_meanings.keys()
    
    cp_meanings = {
        0: 'Typical Angina', 
        1: 'Atypical Angina', 
        2: 'Non-Anginal Pain', 
        3: 'Asymptomatic'
    }
    cp_options = cp_meanings.keys()
    
    fbs_meanings = {0: 'No', 1: 'Yes'}
    fbs_options = fbs_meanings.keys()
    
    restecg_meanings = {
        0: 'Normal', 
        1: 'Having ST-T Wave Abnormality', 
        2: 'Showing Left Ventricular Hypertrophy'
    }
    restecg_options = restecg_meanings.keys()
    
    exang_meanings = {0: 'No', 1: 'Yes'}
    exang_options = exang_meanings.keys()
    
    slope_meanings = {
        0: 'Upsloping', 
        1: 'Flat', 
        2: 'Downsloping'
    }
    slope_options = slope_meanings.keys()
    
    thal_meanings = {
        1: 'Normal', 
        3: 'Fixed Defect', 
        6: 'Reversible Defect', 
        7: 'Unknown'
    }
    thal_options = thal_meanings.keys()

    # Create user input fields
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex', options=list(sex_options), format_func=lambda x: sex_meanings.get(x, x))
    cp = st.sidebar.selectbox('Chest Pain Type', options=list(cp_options), format_func=lambda x: cp_meanings.get(x, x))
    trestbps = st.sidebar.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 100, 400, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', options=list(fbs_options), format_func=lambda x: fbs_meanings.get(x, x))
    restecg = st.sidebar.selectbox('Resting ECG', options=list(restecg_options), format_func=lambda x: restecg_meanings.get(x, x))
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', options=list(exang_options), format_func=lambda x: exang_meanings.get(x, x))
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', options=list(slope_options), format_func=lambda x: slope_meanings.get(x, x))
    ca = st.sidebar.slider('Number of Major Vessels Colored', 0, 4, 0)
    thal = st.sidebar.selectbox('Thalium Stress Test Result', options=list(thal_options), format_func=lambda x: thal_meanings.get(x, x))

    # Create a dictionary of inputs
    data = {
        'age': age,
        'sex': sex,  # Already numeric
        'cp': cp,  # Already numeric
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,  # Already numeric
        'restecg': restecg,  # Already numeric
        'thalach': thalach,
        'exang': exang,  # Already numeric
        'oldpeak': oldpeak,
        'slope': slope,  # Already numeric
        'ca': ca,
        'thal': thal  # Already numeric
    }
    
    # Convert it to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Create the Streamlit app structure
st.title("Heart Disease Prediction App")

st.write("""
### Input patient data:
Use the sidebar to input clinical parameters to predict whether the patient has heart disease or not.
""")

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('User Input Features')
st.write(input_df)

# Model prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
heart_disease_status = np.array(['No Disease', 'Disease'])
st.write(heart_disease_status[prediction][0])

st.subheader('Prediction Probability')
st.write(prediction_proba)
