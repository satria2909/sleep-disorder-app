import joblib
import pandas as pd

bundle = joblib.load("best_sleep_disorder_model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
label_encoder = bundle["label_encoder"]
label_classes = label_encoder.classes_


def predict_sleep(data):

    X = pd.DataFrame([[
        data.Age,
        data.Sleep_Duration,
        data.Quality_of_Sleep,
        data.Physical_Activity_Level,
        data.Stress_Level,
        data.Heart_Rate,
        data.Daily_Steps,
        data.Systolic_BP,
        data.Diastolic_BP

    ]], columns=feature_columns)

    X = scaler.transform(X)

    probabilities = model.predict_proba(X)[0]

    probs = {
        label: round(float(prob) * 100, 2)
        for label, prob in zip(label_classes, probabilities)
    }

    prediction = max(probs, key=probs.get)

    return {
        "prediction": prediction,
        "probability": probs
    }