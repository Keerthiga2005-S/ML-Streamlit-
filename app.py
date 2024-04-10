import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model (1).pkl', 'rb'))

def predict_heart_failure(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, serum_creatinine, serum_sodium, sex, smoking):
    # Convert inputs to float
    input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, serum_creatinine, serum_sodium, sex, smoking]], dtype=np.float64)
    # Predict probability
    prediction = model.predict_proba(input_data)
    pred = '{0:.{1}f}'.format(prediction[0][1], 2)
    return float(pred)

def main():
    st.title('Heart Failure Prediction')
    html_temp = """    
    <div style="background-color:#825246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Heart Failure Prediction ML </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    age = st.text_input("Age")
    anaemia = st.radio("Anaemia", ["Yes", "No"])  # Use radio buttons instead of text input
    creatinine_phosphokinase = st.text_input("Creatinine Phosphokinase")
    diabetes = st.radio("Diabetes", ["Yes", "No"])  # Use radio buttons instead of text input
    ejection_fraction = st.text_input("Ejection Fraction")
    high_blood_pressure = st.radio("High Blood Pressure", ["Yes", "No"])  # Use radio buttons instead of text input
    serum_creatinine = st.text_input("Serum Creatinine")
    serum_sodium = st.text_input("Serum Sodium")
    sex = st.radio("Gender", ["Male", "Female"])  # Use radio buttons instead of text input
    smoking = st.radio("Smoking", ["Yes", "No"])  # Use radio buttons instead of text input
    
    if st.button("Predict"):
        try:
            # Ensure inputs are float
            age = float(age)
            creatinine_phosphokinase = float(creatinine_phosphokinase)
            ejection_fraction = float(ejection_fraction)
            serum_creatinine = float(serum_creatinine)
            serum_sodium = float(serum_sodium)
            
            # Map categorical variables to numeric values
            anaemia = 1 if anaemia == "Yes" else 0
            diabetes = 1 if diabetes == "Yes" else 0
            high_blood_pressure = 1 if high_blood_pressure == "Yes" else 0
            sex = 1 if sex == "Male" else 0
            smoking = 1 if smoking == "Yes" else 0
            
            # Predict
            output = predict_heart_failure(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, serum_creatinine, serum_sodium, sex, smoking)
            st.success('The probability of heart failure is {}'.format(output))
            if output > 0.5:
                st.error("Heart failure is predicted")
            else:
                st.success("Heart failure is not predicted")
        except ValueError:
            st.error("Please enter valid numeric values for age, creatinine phosphokinase, ejection fraction, serum creatinine, and serum sodium.")

if __name__ == '__main__':
    main()
