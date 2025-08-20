from flask import Flask,  request, jsonify
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
de_label_encoders = depression_model_data['label_encoders']

di_m_data = diabetic_model_data["model"]
di_s_data = diabetic_model_data["scaler"]

l_m_data = lung_cancer_model_data["model"]
l_s_data = lung_cancer_model_data["scaler"]

m_m_data = migrane_model_data["model"]
m_s_data = migrane_model_data["scaler"]

@app.route("/api/get-depression-predition",methods=["POST"])
def getDepressionPrediction():
     
     data = request.get_json()

     df_input = pd.DataFrame([data])
     for col, le in de_label_encoders.items():
        if col in df_input:
            try:
                df_input[col] = le.transform(df_input[col])
            except ValueError:
                df_input[col] = le.transform([le.classes_[0]])

     scaled_input = pd.DataFrame(de_s_data.transform(df_input), columns=df_input.columns)

   
     prediction = de_m_data.predict(scaled_input)
 
     if prediction[0] == 1:
        return jsonify({"result":"Depression"})
     elif prediction[0] == 0:
        return jsonify({"result":"Non-Depression"})
     

@app.route("/api/get-diabetic-prediction",methods=["POST"])
def getDiabeticPrediction():
    data = request.get_json()

    feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    df_input = pd.DataFrame([[data["Pregnancies"],data["Glucose"],data["BloodPressure"],data["SkinThickness"],data["Insulin"],data["BMI"],data["DiabetesPedigreeFunction"],data["Age"]]],columns=feature_names)
    
    new_sample_scaled = di_s_data.transform(df_input)

    scaled_input = pd.DataFrame(new_sample_scaled, columns=feature_names)
    
    prediction = di_m_data.predict(scaled_input)
    
    if prediction[0] == 1:
        return jsonify({"result":"Diabetic"})
    elif prediction[0] == 0:
        return jsonify({"result":"Non-Diabetic"})


@app.route("/api/get-lung-cancer-prediction",methods=["POST"])
def getLungCancerPrediction():
    data = request.get_json()

    feature_names = ['GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE',
                   'CHRONIC DISEASE','FATIGUE ','ALLERGY ','WHEEZING','ALCOHOL CONSUMING',
                   'COUGHING','SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN']
    
    df_input = pd.DataFrame([[data["GENDER"],data["AGE"],data["SMOKING"],data["YELLOW_FINGERS"],data["ANXIETY"],data["PEER_PRESSURE"],data["CHRONIC DISEASE"],data["FATIGUE"],data["ALLERGY"],data["WHEEZING"],data["ALCOHOL CONSUMING"],data["COUGHING"],data["SHORTNESS OF BREATH"],data["SWALLOWING DIFFICULTY"],data["CHEST PAIN"]]],columns=feature_names)
    
    scaled_input = l_s_data.transform(df_input)

    new_scaled_input = pd.DataFrame(scaled_input, columns=feature_names)

    prediction = l_m_data.predict(new_scaled_input)

    if prediction[0] == 1:
        return jsonify({"result":"Positive-Lung Cancer"})
    elif prediction[0] == 0:
        return jsonify({"result":"No Signs-Lung Cancer"})
    

@app.route("/api/get-migraine-prediction",methods=["POST"])
def getMigrainePrediction():
    data = request.get_json()

    feature_names = [
    "Age","Duration","Frequency","Location","Character","Intensity","Nausea","Vomit","Phonophobia","Photophobia","Visual","Sensory","Dysphasia","Dysarthria","Vertigo","Tinnitus","Hypoacusis","Diplopia","Defect","Ataxia","Conscience","Paresthesia","DPF"
    ]
    
    df_input = pd.DataFrame([[data["Age"],data["Duration"],data["Frequency"],data["Location"],data["Character"],data["Intensity"],data["Nausea"],data["Vomit"],data["Phonophobia"],data["Photophobia"],data["Visual"],data["Sensory"],data["Dysphasia"],data["Dysarthria"],data["Vertigo"],data["Tinnitus"],data["Hypoacusis"],data["Diplopia"],data["Defect"],data["Ataxia"],data["Conscience"],data["Paresthesia"],data["DPF"]]],columns=feature_names)
    
    new_sample_scaled = m_s_data.transform(df_input)

    new_sample_scaled_df = pd.DataFrame(new_sample_scaled, columns=feature_names)

    prediction = m_m_data.predict(new_sample_scaled_df)
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