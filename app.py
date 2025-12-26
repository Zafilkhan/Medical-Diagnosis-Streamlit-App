import streamlit as st 
import pickle
import numpy as np

st.set_page_config(page_title="Medical Diagnosis App",layout="centered")
st.title("🩺 Medical Diagnosis System")
st.write("Predict Diabetes, Heart Disease & Parkinson's")


#-------------------Model loading-------------------#

diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
diabetes_scaler = pickle.load(open("diabetes_scaler.pkl", "rb"))
heart_model = pickle.load(open("heart_model.pkl", "rb"))
parkinsons_model = pickle.load(open("parkinsons_model.pkl", "rb"))

#-----------sidebar----------------------#
disease  = st.sidebar.selectbox(
    "Select Disease",
    ("Diabetes", "Heart Disease", "Parkinson's")


)
## order of clms
# Pregnancies
# Glucose
# BloodPressure
# SkinThickness
# Insulin
# BMI
# DiabetesPedigreeFunction
# Age


if disease=="Diabetes":
    st.subheader("🩸 Diabetes Prediction")
    col1 , col2  = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0)
        glucose = st.number_input("Glucose", 0)
        bp = st.number_input("BloodPressure", 0)
        skin = st.number_input("SkinThickness", 0)
    with col2:
        insulin = st.number_input("Insulin", 0)
        bmi = st.number_input("BMI", 0.0)
        dpf = st.number_input("DiabetesPedigreeFunction", 0.0)
        age = st.number_input("Age", 1)   

    if st.button("Predict Diabetes"):
        input_data = np.array([[
            pregnancies, glucose, bp, skin,
            insulin, bmi, dpf, age
        ]])    
        input_scaled = diabetes_scaler.transform(input_data)
        prediction = diabetes_model.predict(input_scaled)
        
        if prediction[0] == 1:
            st.error("⚠️ Person has Diabetes")
        else:
            st.success("✅ Person does NOT have Diabetes")

if disease == "Heart Disease":
    st.subheader("❤️ Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1)
        sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", min_value=0)
        chol = st.number_input("Cholesterol", min_value=0)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
        restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", min_value=0)
        exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])

    with col3:
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, format="%.2f")
        slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

    if st.button("Predict Heart Disease"):
        input_data = np.array([[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]])

        prediction = heart_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease Detected")



if disease == "Parkinson's":
    st.subheader("🧠 Parkinson's Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", 0.0)
        fhi = st.number_input("MDVP:Fhi(Hz)", 0.0)
        flo = st.number_input("MDVP:Flo(Hz)", 0.0)
        jitter_percent = st.number_input("MDVP:Jitter(%)", 0.0)
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0)
        rap = st.number_input("MDVP:RAP", 0.0)
        ppq = st.number_input("MDVP:PPQ", 0.0)

    with col2:
        ddp = st.number_input("Jitter:DDP", 0.0)
        shimmer = st.number_input("MDVP:Shimmer", 0.0)
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.0)
        apq3 = st.number_input("Shimmer:APQ3", 0.0)
        apq5 = st.number_input("Shimmer:APQ5", 0.0)
        apq = st.number_input("MDVP:APQ", 0.0)
        dda = st.number_input("Shimmer:DDA", 0.0)

    with col3:
        nhr = st.number_input("NHR", 0.0)
        hnr = st.number_input("HNR", 0.0)
        rpde = st.number_input("RPDE", 0.0)
        dfa = st.number_input("DFA", 0.0)
        spread1 = st.number_input("spread1", 0.0)
        spread2 = st.number_input("spread2", 0.0)
        d2 = st.number_input("D2", 0.0)
        ppe = st.number_input("PPE", 0.0)

    if st.button("Predict Parkinson's Disease"):
        input_data = np.array([[
            fo,
            fhi,
            flo,
            jitter_percent,
            jitter_abs,
            rap,
            ppq,
            ddp,
            shimmer,
            shimmer_db,
            apq3,
            apq5,
            apq,
            dda,
            nhr,
            hnr,
            rpde,
            dfa,
            spread1,
            spread2,
            d2,
            ppe
        ]])

        prediction = parkinsons_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Parkinson's Disease Detected")
        else:
            st.success("✅ No Parkinson's Disease Detected")
