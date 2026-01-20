# src/verification.py
import pickle

VERIFICATION_MODEL_PATH = "models/verification_model.pkl"

def load_verification_model():
    with open(VERIFICATION_MODEL_PATH, "rb") as f:
        return pickle.load(f)

def verify_document(doc, model, threshold=0.6):
    features = [[
        float(doc.get("trust", 0.5)),
        int(doc.get("source_type", "").lower() == "official"),
        int("panic" in doc.get("text", "").lower()),
        int(doc.get("source_type", "").lower() in ["official", "reliable"])
    ]]

    prob = model.predict_proba(features)[0][1]
    return prob >= threshold, prob
