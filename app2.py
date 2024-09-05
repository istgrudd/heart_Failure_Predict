import numpy as np
import pickle
import streamlit as st
import sklearn

# Load the model and scaler
with open('model_classifier.pkl', 'rb') as classifier_file:
    classifier = pickle.load(classifier_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to make prediction based on user input 
def prediction(Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope):
    # Label mappings
    sex_mapping = {'Female': 0, 'Male': 1}
    Sex = sex_mapping[Sex]

    chestpain_mapping = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
    ChestPainType = chestpain_mapping[ChestPainType]

    restingecg_mapping = {'LVH': 0, 'Normal': 1, 'ST': 2}
    RestingECG = restingecg_mapping[RestingECG]
    
    exercise_mapping = {'N': 0, 'Y': 1}
    ExerciseAngina = exercise_mapping[ExerciseAngina]
    
    st_slope_mapping = {'Down': 0, 'Flat': 1, 'Up': 2}
    ST_Slope = st_slope_mapping[ST_Slope]

    # Feature scaling for numeric inputs
    ncols = np.array([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])
    ncols_scaled = scaler.transform(ncols)

    # Concatenate scaled numeric inputs with categorical ones
    input_data = np.concatenate([ncols_scaled], axis=1)

    # Make predictions based on probabilities
    prediction_proba = classifier.predict_proba(input_data)  # Get the probability of each class

    # Label mapping for prediction output
    heartdisease_types = {1: 'Normal', 0: 'Heart Disease'}

    # Create a list of (Heart Disease, probability) tuples
    predictions = [(heartdisease_types[i], prediction_proba[0][i]) for i in range(len(heartdisease_types))]

    # Sort by probability, highest first
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions

# Main function for webpage
def main():
    # Page config
    st.set_page_config(page_title="Heart Disease Prediction App", page_icon="💓", layout="centered")

    # Web title
    st.markdown("<div class='main-title'>Heart Disease Prediction ML App</div>", unsafe_allow_html=True)

    # Input container
    with st.form("prediction_form"):
        # Input fields
        Age = st.slider("Age (Year)", min_value=28, max_value=77)
        RestingBP = st.slider("Resting BP (mmHg)", min_value=0, max_value=200)
        Cholesterol = st.slider("Cholesterol (mg/dL)", min_value=0, max_value=603)
        MaxHR = st.slider("Max HR (BPM)", min_value=60, max_value=202)
        Oldpeak = st.slider("Oldpeak", min_value=-2.6, max_value=6.2, step=0.1)
        
        Sex = st.selectbox('Sex', ('Female', 'Male'))
        ChestPainType = st.selectbox('Chest Pain Type', ('ASY', 'ATA', 'NAP', 'TA'))
        FastingBS = st.selectbox('Fasting BS', ('0', '1'))
        RestingECG = st.selectbox('Resting ECG', ('LVH', 'Normal', 'ST'))
        ExerciseAngina = st.selectbox('Exercise Angina', ('N', 'Y'))
        ST_Slope = st.selectbox('ST Slope', ('Down', 'Flat', 'Up'))

        # Predict button
        submitted = st.form_submit_button("Predict")
        st.markdown("</div>", unsafe_allow_html=True)

    # Make the prediction and display results when 'Predict' is clicked
    if submitted:
        predictions = prediction(Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope)

        # Display the probabilities with appropriate background color
        for i, (disease_status, prob) in enumerate(predictions):
            if i == 0:  # highest probability
                st.markdown(f"<div class='result green'>The prediction is {disease_status} with a confidence of {prob*100:.2f}%</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result yellow'>The prediction is {disease_status} with a confidence of {prob*100:.2f}%</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
