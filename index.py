from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)

# Load models
def load_model(path):
    return joblib.load(os.path.join(BASE_DIR, "../model", path))

depression_model_data = load_model("depression_best_model.pkl")
diabetic_model_data = load_model("diabetic_best_model.pkl")
lung_model_data = load_model("lung_cancer_best_model.pkl")
migraine_model_data = load_model("migrane_best_model.pkl")

# Extract parts
de_m = depression_model_data["model"]
de_s = depression_model_data["scaler"]
de_enc = depression_model_data["label_encoders"]

di_m = diabetic_model_data["model"]
di_s = diabetic_model_data["scaler"]

l_m = lung_model_data["model"]
l_s = lung_model_data["scaler"]

m_m = migraine_model_data["model"]
m_s = migraine_model_data["scaler"]

# ------------------ ROUTES ------------------

@app.route("/api/get-depression-predition", methods=["POST"])
def depression():
    data = request.get_json()
    df = pd.DataFrame([data])

    for col, le in de_enc.items():
        if col in df:
            try:
                df[col] = le.transform(df[col])
            except:
                df[col] = le.transform([le.classes_[0]])

    scaled = de_s.transform(df)
    pred = de_m.predict(scaled)

    return jsonify({"result": "Depression" if pred[0] == 1 else "Non-Depression"})


@app.route("/api/get-diabetic-prediction", methods=["POST"])
def diabetic():
    data = request.get_json()

    features = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                "Insulin","BMI","DiabetesPedigreeFunction","Age"]

    df = pd.DataFrame([[data[f] for f in features]], columns=features)

    scaled = di_s.transform(df)
    pred = di_m.predict(scaled)

    return jsonify({"result": "Diabetic" if pred[0] == 1 else "Non-Diabetic"})


@app.route("/api/get-lung-cancer-prediction", methods=["POST"])
def lung():
    data = request.get_json()

    features = ['GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE',
                'CHRONIC DISEASE','FATIGUE','ALLERGY','WHEEZING','ALCOHOL CONSUMING',
                'COUGHING','SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN']

    df = pd.DataFrame([[data[f] for f in features]], columns=features)

    scaled = l_s.transform(df)
    pred = l_m.predict(scaled)

    return jsonify({"result": "Positive-Lung Cancer" if pred[0] == 1 else "No Signs-Lung Cancer"})


@app.route("/api/get-migraine-prediction", methods=["POST"])
def migraine():
    data = request.get_json()

    features = ["Age","Duration","Frequency","Location","Character","Intensity",
                "Nausea","Vomit","Phonophobia","Photophobia","Visual","Sensory",
                "Dysphasia","Dysarthria","Vertigo","Tinnitus","Hypoacusis",
                "Diplopia","Defect","Ataxia","Conscience","Paresthesia","DPF"]

    df = pd.DataFrame([[data[f] for f in features]], columns=features)

    scaled = m_s.transform(df)
    pred = m_m.predict(scaled)

    results = {
        0: "Migraine without aura",
        1: "Typical aura without migraine",
        2: "Migraine without aura",
        3: "Familial hemiplegic migraine",
        4: "Sporadic hemiplegic migraine",
        5: "Typical aura with migraine",
        6: "Basilar-type aura"
    }

    return jsonify({"result": results.get(pred[0], "Unknown")})


# 👇 REQUIRED for Vercel
# DO NOT use app.run()