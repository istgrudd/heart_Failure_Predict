import numpy as np
import pickle
import streamlit as st
import sklearn

with open('model_classifier.pkl', 'rb') as classifier_file:
    classifier = pickle.load(classifier_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# function to make prediction based on user input 
def prediction(Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease):
    # label mapping
    sex_mapping = {'F': 0, 'M': 1}
    Sex = sex_mapping[Sex]

    chestpain_mapping = {'ATA': 1, 'NAP': 2, 'ASY': 0, 'TA': 3}
    ChestPainType = season_mapping[ChestPainType]

    restingecg_mapping = {'Normal': 1, 'ST': 2, 'LVH': 0}
    RestingECG = restingecg_mapping[RestingECG]
    
    exercise_mapping = {'N': 0, 'Y': 1}
    ExerciseAngina = exercise_mapping[ExerciseAngina]
    
    restingecg_mapping = {'Flat': 1, 'Up': 2, 'Down': 0}
    Location = location_mapping[Location]    

    # feature scaling
    num_cols = np.array([[Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak]])
    num_cols_scaled = scaler.transform(num_cols)
    input_data = np.concatenate([num_cols_scaled, [[Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope]]], axis=1)

    # make predictions based on probabilities
    prediction_proba = classifier.predict_proba(input_data)  # Get the probability of each class

    # label mapping
    heartdisease_types = {0: 'Normal', 1: 'Heart Disease'}
    HeartDisease = heartdisease_types[HeartDisease]

    # create a list of (weather, probability) tuples
    predictions = [(heartdisease_types[i], prediction_proba[0][i]) for i in range(len(heartdisease_types))]

    # sort by probability, highest first
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions

# main functiofor webpage
def main():
    # page config
    st.set_page_config(page_title="Heart Disease Prediction App", page_icon="⛅", layout="centered")

    # web-title
    st.markdown("<div class='main-title'>Weather Prediction ML App</div>", unsafe_allow_html=True)

    # input container
    with st.form("prediction_form"):

        # input fields
        Age = st.number_input("Age (Year)", value=None, step=1.0, placeholder='Enter the Age in Year')
        RestingBP = st.number_input("Resting BP (mmHg)", value=None, step=1.0, placeholder='Enter the Resting Blood Pressure in mmHg')
        Cholesterol = st.number_input("Cholesterol (mg/dL)", value=None, step=1.0, placeholder='Enter the Cholesterol in mg/dL')
        MaxHR = st.number_input("Max HR (BPM)", value=None, step=1.0, placeholder='Enter the MAx Heart Rate in BPM')
        Oldpeak = st.slider("Oldpeak", min_value=-2.6, max_value=6.2, step=0.1)
        
        Sex = st.selectbox('Sex', ('Female', 'Male'))
        ChestPainType = st.selectbox('Chest Paint Type', ('ASY', 'ATA', 'NAP', 'TA'))
        FastingBS = st.selectbox('Fasting BS', ('0', '1'))
        RestingECG = st.selectbox('Resting ECG', ('LVH', 'Normal', 'ST'))
        ExerciseAngina = st.selectbox('Exercise Angina', ('N', 'Y'))
        ST_Slope = st.selectbox('ST Slope', ('Down', 'Flat', 'Up'))

        # predict button
        submitted = st.form_submit_button("Predict")
        st.markdown("</div>", unsafe_allow_html=True)

    # make the prediction and display results when 'Predict' is clicked
    if submitted:
        predictions = prediction(Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease)

        # display the probabilities with appropriate background color
        for i, (weather, prob) in enumerate(predictions):
            if i == 0:  # highest probability
                st.markdown(f"<div class='result green'>The weather is {weather} with a confidence of {prob*100:.2f}%</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result yellow'>The weather is {weather} with a confidence of {prob*100:.2f}%</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()