import pickle
import numpy as np
import os

model_path = "model/model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ model.pkl file not found. Please run train_model.py first.")

with open(model_path, "rb") as f:
    try:
        model = pickle.load(f)
    except EOFError:
        raise Exception("❌ model.pkl is empty or corrupted. Please retrain your model.")

def predict_readmission(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]
    return prediction, probability
