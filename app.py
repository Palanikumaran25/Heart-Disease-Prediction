import pandas as pd
import numpy as np 
import streamlit as st 
import pickle
import os 
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Heart Disease Predict",
                   layout="wide",
                   page_icon="🧑‍⚕️") 

working_dir = os.path.dirname(os.path.abspath(__file__))

Heart_model = pd.read_csv(''C:\Python\Heart Disease Prediction\heart (1).csv'')


models = { 
        "Select a model": None,
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier()
} 

#Sidebar: Model Selection
st.sidebar.header("⚙️ Model Settings")
selected_model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))
selected_model = models[selected_model_name] 


#sidebar for navigate 
with st.sidebar: 
    selected = option_menu ("Disease Prediction:",
                            ["Heart Disease Prediction"],
                            icons=['heart'],
                            default_index =0)  
 
with st.sidebar:   
    st.subheader("📁 Upload CSV File:")
    uploaded_file = st.file_uploader("Heart disease csv file", type=["csv"])
    
    if uploaded_file is not None: 
        df_upload = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data")
        st.dataframe(df_upload)

   
if (selected == 'Heart Disease Prediction'): 
    st.title("Heart Prediction using ML Model:") 
    
    col1, col2, col3 = st.columns(3) 
    
    with col1: 
        age = st.text_input('Age :')
        
    with col2: 
        sex = st.text_input('Sex :')
        
    with col3: 
        cp = st.text_input('Chest Pain types :')
        
    with col1: 
        trestbps = st.text_input('Resting Blood Pressure :')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl :')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl :')
        
    with col1: 
        restecg = st.text_input('Resting Electrocardiographic results :')
        
    with col2: 
        thalach = st.text_input('Maximum Heart Rate achieved :')
        
    with col3: 
        exang = st.text_input('Exercise Induced Angina :')
        
    with col1: 
        oldpeak = st.text_input('ST depression induced by exercise :')
        
    with col2: 
        slope = st.text_input('Slope of the peak exercise ST segment :')
        
    with col3: 
        ca = st.text_input('Major vessels colored by flourosopy :')
        
    with col1: 
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect :')

    heart_diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = Heart_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)
    


