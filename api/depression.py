import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
model_data = joblib.load(os.path.join(BASE_DIR, "../model/depression_best_model.pkl"))

model = model_data["model"]
scaler = model_data["scaler"]
label_encoders = model_data["label_encoders"]

def handler(request):
    data = request.get_json()

    df_input = pd.DataFrame([data])

    for col, le in label_encoders.items():
        if col in df_input:
            try:
                df_input[col] = le.transform(df_input[col])
            except:
                df_input[col] = le.transform([le.classes_[0]])

    scaled = scaler.transform(df_input)
    prediction = model.predict(scaled)

    return {
        "statusCode": 200,
        "headers": {"Access-Control-Allow-Origin": "*"},
        "body": {"result": "Depression" if prediction[0] == 1 else "Non-Depression"}
    }