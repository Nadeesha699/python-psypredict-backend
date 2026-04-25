import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
model_data = joblib.load(os.path.join(BASE_DIR, "../model/diabetic_best_model.pkl"))

model = model_data["model"]
scaler = model_data["scaler"]

def handler(request):
    data = request.get_json()

    feature_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    df_input = pd.DataFrame([[ 
        data["Pregnancies"], data["Glucose"], data["BloodPressure"],
        data["SkinThickness"], data["Insulin"], data["BMI"],
        data["DiabetesPedigreeFunction"], data["Age"]
    ]], columns=feature_names)

    scaled = scaler.transform(df_input)
    prediction = model.predict(scaled)

    return {
        "statusCode": 200,
        "headers": {"Access-Control-Allow-Origin": "*"},
        "body": {"result": "Diabetic" if prediction[0] == 1 else "Non-Diabetic"}
    }