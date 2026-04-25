import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
model_data = joblib.load(os.path.join(BASE_DIR, "../model/migrane_best_model.pkl"))

model = model_data["model"]
scaler = model_data["scaler"]

def handler(request):
    data = request.get_json()

    feature_names = [
        "Age","Duration","Frequency","Location","Character","Intensity",
        "Nausea","Vomit","Phonophobia","Photophobia","Visual","Sensory",
        "Dysphasia","Dysarthria","Vertigo","Tinnitus","Hypoacusis",
        "Diplopia","Defect","Ataxia","Conscience","Paresthesia","DPF"
    ]

    df_input = pd.DataFrame([[ 
        data["Age"], data["Duration"], data["Frequency"], data["Location"],
        data["Character"], data["Intensity"], data["Nausea"], data["Vomit"],
        data["Phonophobia"], data["Photophobia"], data["Visual"], data["Sensory"],
        data["Dysphasia"], data["Dysarthria"], data["Vertigo"], data["Tinnitus"],
        data["Hypoacusis"], data["Diplopia"], data["Defect"], data["Ataxia"],
        data["Conscience"], data["Paresthesia"], data["DPF"]
    ]], columns=feature_names)

    scaled = scaler.transform(df_input)
    prediction = model.predict(scaled)

    results = {
        0: "Migraine without aura",
        1: "Typical aura without migraine",
        2: "Migraine without aura",
        3: "Familial hemiplegic migraine",
        4: "Sporadic hemiplegic migraine",
        5: "Typical aura with migraine",
        6: "Basilar-type aura"
    }

    return {
        "statusCode": 200,
        "headers": {"Access-Control-Allow-Origin": "*"},
        "body": {"result": results.get(prediction[0], "Unknown")}
    }