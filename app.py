from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

depression_model_data = joblib.load("./model/depression_best_model.pkl")
diabetic_model_data = joblib.load("./model/diabetic_best_model.pkl")
lung_cancer_model_data = joblib.load("./model/lung_cancer_best_model.pkl")
migrane_model_data = joblib.load("./model/migrane_best_model.pkl")

de_m_data = depression_model_data["model"]
de_s_data = depression_model_data["scaler"]

di_m_data = diabetic_model_data["model"]
di_s_data = diabetic_model_data["scaler"]

l_m_data = lung_cancer_model_data["model"]
l_s_data = lung_cancer_model_data["scaler"]

m_m_data = migrane_model_data["model"]
m_s_data = migrane_model_data["scaler"]

@app.route("/api/get-depression-predition",methods=["POST"])
def getDepressionPrediction():
     
     data = request.get_json()
     encoded_input = {
      "Gender": 1 if data["Gender"] == "Male" else 0,
      "Age": data["Age"],
      "Profession": 0 if data["Profession"] == "Student" else 1, 
      "Academic Pressure": data["Academic Pressure"],
      "Work Pressure": data["Work Pressure"],
      "CGPA": data["CGPA"],
      "Study Satisfaction": data["Study Satisfaction"],
      "Job Satisfaction": data["Job Satisfaction"],
      "Sleep Duration": 1 if data["Sleep Duration"] == "5-6 hours" else 0, 
      "Dietary Habits": 1 if data["Dietary Habits"] == "Healthy" else 0,
      "Have you ever had suicidal thoughts ?": 1 if data["Have you ever had suicidal thoughts ?"] == "Yes" else 0,
      "Work/Study Hours": data["Work/Study Hours"],
      "Financial Stress": data["Financial Stress"],
      "Family History of Mental Illness": 1 if data["Family History of Mental Illness"] == "Yes" else 0
     }

     df_input = pd.DataFrame([encoded_input])
     scaled_input = de_s_data.transform(df_input)
     prediction = de_m_data.predict(scaled_input)
     if prediction[0] == 1:
        return jsonify({"result":"Depression"})
     elif prediction[0] == 0:
        return jsonify({"result":"Non-Depression"})
     

@app.route("/api/get-diabetic-prediction",methods=["POST"])
def getDiabeticPrediction():
    data = request.get_json()
    df_input = np.array([[data["Pregnancies"],data["Glucose"],data["BloodPressure"],data["SkinThickness"],data["Insulin"],data["BMI"],data["DiabetesPedigreeFunction"],data["Age"]]])
    scaled_input = di_s_data.transform(df_input)
    prediction = di_m_data.predict(scaled_input)
    if prediction[0] == 1:
        return jsonify({"result":"Diabetic"})
    elif prediction[0] == 0:
        return jsonify({"result":"Non-Diabetic"})


@app.route("/api/get-lung-cancer-prediction",methods=["POST"])
def getLungCancerPrediction():
    data = request.get_json()
    df_input = np.array([[data["GENDER"],data["AGE"],data["SMOKING"],data["YELLOW_FINGERS"],data["ANXIETY"],data["PEER_PRESSURE"],data["CHRONIC DISEASE"],data["FATIGUE"],data["ALLERGY"],data["WHEEZING"],data["ALCOHOL CONSUMING"],data["COUGHING"],data["SHORTNESS OF BREATH"],data["SWALLOWING DIFFICULTY"],data["CHEST PAIN"]]])
    scaled_input = l_s_data.transform(df_input) 
    prediction = l_m_data.predict(scaled_input) 
    if prediction[0] == 1:
        return jsonify({"result":"Positive for Lung Cancer"})
    elif prediction[0] == 0:
        return jsonify({"result":"No Signs of Lung Cancer Detected"})
    

@app.route("/api/get-migraine-prediction",methods=["POST"])
def getMigrainePrediction():
    data = request.get_json()
    df_input =np.array([[data["Age"],data["Duration"],data["Frequency"],data["Location"],data["Character"],data["Intensity"],data["Nausea"],data["Vomit"],data["Phonophobia"],data["Photophobia"],data["Visual"],data["Sensory"],data["Dysphasia"],data["Dysarthria"],data["Vertigo"],data["Tinnitus"],data["Hypoacusis"],data["Diplopia"],data["Defect"],data["Ataxia"],data["Conscience"],data["Paresthesia"],data["DPF"]]])
    scaled_input = m_s_data.transform(df_input)
    prediction = m_m_data.predict(scaled_input)
    if prediction[0] == 0:
        return jsonify({"result":"Migraine without aura"})
    elif prediction[0] == 1:
        return jsonify({"result":"Typical auro without migraine"})
    elif prediction[0] == 2:
        return jsonify({"result":"Migraine without aura"})
    elif prediction[0] == 3:
        return jsonify({"result":"Familial hemiplegic migraine"})
    elif prediction[0] == 4:
        return jsonify({"result":"Sporadic hemiplegic migraine"})
    elif prediction[0] == 5:
        return jsonify({"result":"Typical aura with migraine"})

if __name__ == "__main__":
    app.run(debug=True)