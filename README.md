# 🩺 Medical Diagnosis System using Machine Learning

This project is a complete **end-to-end Machine Learning application** that helps in predicting three common diseases — **Diabetes, Heart Disease, and Parkinson’s Disease**.  
The idea behind this project was not just to train ML models, but to understand the **full workflow**, starting from data preprocessing and statistical understanding, to model training, and finally deploying everything as a **working web application using Streamlit**.

---

## 🔍 Why I built this project

Early detection of diseases can help doctors and patients take timely action.  
As a student of **AI & Machine Learning**, I wanted to build a practical project where I could apply my ML knowledge to a real-world problem and also learn how to deploy models instead of keeping them limited to Jupyter notebooks.

---

## 🩸 Diabetes Prediction – My Approach

For the diabetes dataset, I followed a proper ML workflow.

First, I explored the dataset to understand the medical features such as glucose level, BMI, insulin, age, etc. Since all these features were on different scales, I applied **StandardScaler** to normalize the data.  
After scaling, I analyzed some basic statistical information like mean and standard deviation to understand how scaling improved feature distribution.

For model training, I used **Support Vector Classifier (SVC)** because it performs well on medical datasets and works effectively with scaled data.  
Once the model was trained and tested, I saved both the **trained SVC model and the fitted StandardScaler** using pickle so that the same preprocessing could be applied during prediction time.

---

## ❤️ Heart Disease Prediction – My Approach

For heart disease prediction, I worked with features such as age, chest pain type, cholesterol level, blood pressure, and heart rate.

In this case, I used **Logistic Regression**, as it is simple, interpretable, and well-suited for binary classification problems.  
Since the feature values were already in a reasonable numerical range and Logistic Regression handled them well, I did **not apply feature scaling** here.

The trained Logistic Regression model predicts whether a person is likely to have heart disease or not based on the given medical parameters.

---

## 🧠 Parkinson’s Disease Prediction – My Approach

For Parkinson’s disease, the dataset was based on **biomedical voice measurements** such as jitter, shimmer, frequency, and noise-to-harmonic ratios.

Before training, I removed non-useful columns like `name` and used the `status` column as the target variable.  
The model was trained on these voice-based features to classify whether Parkinson’s disease is present or not.  
The trained model was then saved and later used directly in the Streamlit application for real-time predictions.

---

## 📓 Model Training

All three models were trained and evaluated in **Jupyter Notebook**.  
This helped me clearly understand data preprocessing, feature behavior, and model performance before moving to deployment.  
After training, the models were saved as `.pkl` files so they could be easily loaded into the web application.

---

## 🌐 Web Application & Deployment

To make the project interactive and user-friendly, I built a **Streamlit web application** where users can enter medical details and get instant predictions.

The complete project is:
- Version-controlled using **Git and GitHub**
- Deployed using **Streamlit Community Cloud**
- Accessible through a browser without any local setup

---

## 🛠️ Technologies Used

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Streamlit  
- Jupyter Notebook  
- Git & GitHub  

---

## 📂 Project Structure

Medical-Diagnosis-Streamlit-App/
│
- ├── app.py
- ├── diabetes_model.pkl
- ├── diabetes_scaler.pkl
- ├── heart_model.pkl
- ├── parkinsons_model.pkl
- ├── requirements.txt
- ├── README.md


---

## ▶️ How to Run the Project Locally

1. Clone the repository:
```bash
git clone https://github.com/Zafilkhan/Medical-Diagnosis-Streamlit-App.git
cd Medical-Diagnosis-Streamlit-App
