import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
model_data = joblib.load(os.path.join(BASE_DIR, "../model/lung_cancer_best_model.pkl"))

model = model_data["model"]
scaler = model_data["scaler"]

def handler(request):
    data = request.get_json()

    feature_names = [
        'GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE',
        'CHRONIC DISEASE','FATIGUE ','ALLERGY ','WHEEZING','ALCOHOL CONSUMING',
        'COUGHING','SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN'
    ]

    df_input = pd.DataFrame([[ 
        data["GENDER"], data["AGE"], data["SMOKING"], data["YELLOW_FINGERS"],
        data["ANXIETY"], data["PEER_PRESSURE"], data["CHRONIC DISEASE"],
        data["FATIGUE"], data["ALLERGY"], data["WHEEZING"],
        data["ALCOHOL CONSUMING"], data["COUGHING"],
        data["SHORTNESS OF BREATH"], data["SWALLOWING DIFFICULTY"],
        data["CHEST PAIN"]
    ]], columns=feature_names)

    scaled = scaler.transform(df_input)
    prediction = model.predict(scaled)

    return {
        "statusCode": 200,
        "headers": {"Access-Control-Allow-Origin": "*"},
        "body": {"result": "Positive-Lung Cancer" if prediction[0] == 1 else "No Signs-Lung Cancer"}
    }