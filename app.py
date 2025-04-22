import pickle
import requests
import streamlit as st
from streamlit_option_menu import option_menu
import openai
import pandas as pd
import numpy as np
import json
import base64
import os

model_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
with open(model_path, "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(
    page_title="Healthcare System",
    page_icon=":health_worker:",
    layout="centered",
    initial_sidebar_state="expanded"
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;, height: 0;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# loading the saved models


diabetes_model = pickle.load(open("Diabetes Model.pkl", 'rb'))

heart_disease_model = pickle.load(open("Heart Disease Model.pkl",'rb'))

liver_model = pickle.load(open("Liver Disease Model.pkl", 'rb'))

scaler = pickle.load(open("scaler.pkl", 'rb'))

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Healthcare System',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Liver Disease Prediction',
                           'Healthcare Chatbot'],
                          icons=['droplet-fill','heart','person'],
                          default_index=0)
    
    
if selected == 'Diabetes Prediction':
    # Page title
    st.title('Diabetes Prediction')
    st.markdown("Note: 1: Gender (Female: 0, Male: 1)")
    st.markdown("Note: 2: Smoking History (never: 0, No Info: 1, current: 2, former:3, ever:4, not current: 5)")
    st.markdown("Note: 3: Heart Disease (No: 0 , Yes: 1)")
    # Input fields
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8 = st.columns(2)

    with col1:
        gender = st.text_input('Gender ')
        gender = float(gender) if gender else 0.0

    with col2:
        age = st.text_input('Age')
        age = float(age) if age else 0.0

    with col3:
        hypertension = st.text_input('Hypertension Value')
        hypertension = float(hypertension) if hypertension else 0.0

    with col4:
        heart_disease = st.text_input('Heart Disease')
        heart_disease = float(heart_disease) if heart_disease else 0.0

    with col5:
        smoking_history = st.text_input('Smoking_history Level')
        smoking_history = float(smoking_history) if smoking_history else 0.0

    with col6:
        BMI = st.text_input('BMI value')
        BMI = float(BMI) if BMI else 0.0

    with col7:
        HbA1c_level = st.text_input('HbA1c_level value')
        HbA1c_level = float(HbA1c_level) if HbA1c_level else 0.0

    with col8:
        blood_glucose_level = st.text_input('Blood Glucose Level')
        blood_glucose_level = float(blood_glucose_level) if blood_glucose_level else 0.0

    # Perform prediction
    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        input_data = np.array([[gender, age, hypertension, heart_disease, smoking_history, BMI, HbA1c_level, blood_glucose_level]], dtype=object)
        input_data = input_data.astype(float)

        diab_prediction = diabetes_model.predict(input_data)

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is predicted to have diabetes.'
            st.error(diab_diagnosis)
        else:
            diab_diagnosis = 'The person is predicted to be healthy.'
            st.success(diab_diagnosis)

    



# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction')
    st.markdown("Note: 1: Gender (Female: 0, Male: 1)")
    st.markdown("Note: 2: Thal (Normal: 0, Fixed Defect: 1, Reversible Defect: 2)")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9 = st.columns(3)
    col10, col11, col12, col13 = st.columns(4)

    # Input fields
    with col1:
        age = st.text_input('Age')
        age = float(age) if age else 0.0

    with col2:
        gender = st.text_input('Gender')
        gender = float(gender) if gender else 0.0

    with col3:
        cp = st.text_input('Chest Pain types')
        cp = float(cp) if cp else 0.0

    with col4:
        trestbps = st.text_input('Resting Blood Pressure')
        trestbps = float(trestbps) if trestbps else 0.0

    with col5:
        chol = st.text_input('Serum Cholestoral')
        chol = float(chol) if chol else 0.0

    with col6:
        fbs = st.text_input('Fasting Blood Sugar')
        fbs = float(fbs) if fbs else 0.0

    with col7:
        restecg = st.text_input('Resting Electrocardiographic results')
        restecg = float(restecg) if restecg else 0.0

    with col8:
        thalach = st.text_input('Maximum Heart Rate achieved')
        thalach = float(thalach) if thalach else 0.0

    with col9:
        exang = st.text_input('Exercise Induced Angina')
        exang = float(exang) if exang else 0.0

    with col10:
        oldpeak = st.text_input('ST depression induced by exercise')
        oldpeak = float(oldpeak) if oldpeak else 0.0

    with col11:
        slope = st.text_input('Slope of the peak exercise ST segment')
        slope = float(slope) if slope else 0.0

    with col12:
        ca = st.text_input('Major vessels colored by fluoroscopy')
        ca = float(ca) if ca else 0.0

    with col13:
        thal = st.text_input('Thal Value')
        thal = float(thal) if thal else 0.0

    # Perform prediction
    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        input_data = np.array([[age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], dtype=object)
        input_data = input_data.astype(float)

        heart_prediction = heart_disease_model.predict(input_data)

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is predicted to have heart disease.'
            st.error(heart_diagnosis)
        else:
            heart_diagnosis = 'The person is predicted to be healthy.'
            st.success(heart_diagnosis)

    
        
    
    

# Liver Prediction Page
if (selected == "Liver Disease Prediction"):
    
    # page title
    st.title("Liver Disease Prediction")
    st.markdown("Note: 1: Gender (Female: 0, Male: 1)")
    
    col1, col2, col3 = st.columns(3)  
    col4, col5, col6 = st.columns(3)
    
    with col1:
        Age = st.text_input('Age')
        
    with col2:
        gender = st.text_input('Gender')
        
    with col3:
        Total_Bilirubin = st.text_input('Total Bilirubin')
        
    with col4:
        Alkaline_Phosphotase = st.text_input('Alkaline Phosphotase')
        
    with col5:
        Alamine_Aminotransferase = st.text_input('Alamine Aminotransferase')

    with col6:
        Albumin_and_Globulin_Ratio = st.text_input('Albumin and Globulin Ratio')
        
    
    # code for Prediction
    liver_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Liver Test Result"):
        def preprocess_input(data):
            # Apply log1p transformation
            skewed = ['Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase','Albumin_and_Globulin_Ratio']

            data[skewed] = np.log1p(data[skewed])

            # Scale the data using the loaded scaler
            attributes = [col for col in data.columns]
            data[attributes] = scaler.transform(data[attributes])

            return data
        
        input_data = [Age,gender,Total_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Albumin_and_Globulin_Ratio]  
        column_names = ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_Phosphotase','Alamine_Aminotransferase', 'Albumin_and_Globulin_Ratio']
        # Convert the user's input into a pandas DataFrame
        user_data = pd.DataFrame([input_data], columns=column_names)  
        user_data[column_names] = user_data[column_names].apply(pd.to_numeric, errors='coerce')
        # Preprocess the user's input data
        preprocessed_data = preprocess_input(user_data)   

        prediction = liver_model.predict(preprocessed_data)                  
        
        if (prediction[0] == 0):
          liver_diagnosis = "The person does not have a Liver disease"
          st.success(liver_diagnosis)
        else:
          liver_diagnosis = "The Person has Liver Disease"
          st.error(liver_diagnosis)
    

if selected == 'Healthcare Chatbot':
    # Set Groq API key and endpoint
    GROQ_API_KEY = "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"  # 🔑 Replace with your actual key
    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    # Function to generate a response from Groq API
    def generate_response(prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        data = {
            "model": "mistral-saba-24b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        response = requests.post(API_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"

    # ✅ Base64 conversion function for image
    def get_base64_image(image_path):
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            return f"data:image/png;base64,{base64.b64encode(data).decode()}"
        except FileNotFoundError:
            return None

    # Title and input
    st.title("Healthcare Chatbot")
    st.markdown("Welcome to the **Healthcare Chatbot**! How can I assist you today?")

    user_input = st.text_input("User:", placeholder="Type your health query here...")

    if user_input:
        with st.spinner("Bot is typing..."):
            response = generate_response(user_input)
        st.markdown(f"**Bot:** {response}")

    # Floating chatbot image animation (bottom-right)
    img_base64 = get_base64_image("chatbot-bot.png")  # Ensure image is in same directory
    if img_base64:
        st.markdown(f"""
            <style>
                .chatbot-image {{
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    z-index: 9999;
                    animation: floatBot 2s ease-in-out infinite;
                }}
                @keyframes floatBot {{
                    0% {{ transform: translateY(0); }}
                    50% {{ transform: translateY(-10px); }}
                    100% {{ transform: translateY(0); }}
                }}
            </style>
            <img src="{img_base64}" class="chatbot-image" width="300" height="300">
        """, unsafe_allow_html=True)
    else:
        st.warning("Chatbot image not found. Make sure 'chatbot-bot.png' is in the same folder as your app.")