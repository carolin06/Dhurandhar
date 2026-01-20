# src/intent_inference.py
import pickle

INTENT_MODEL_PATH = "models/intent_model.pkl"

# -------------------------------
# EXPANDED KEYWORD DICTIONARIES
# -------------------------------

DISASTER_WORDS = [
    "earthquake", "quake", "aftershock",
    "flood", "flooding", "flash flood",
    "cyclone", "hurricane", "typhoon",
    "tsunami", "tidal wave",
    "landslide", "mudslide",
    "volcano", "eruption",
    "wildfire", "forest fire", "bushfire",
    "drought", "avalanche",
    "heatwave", "cold wave",
    "storm", "thunderstorm",
    "fire", "explosion", "blast",
    "chemical leak", "gas leak",
    "building collapse", "collapse",
    "train accident", "plane crash",
    "road accident", "stampede"
]

ACTION_WORDS = [
    "evacuate", "evacuation",
    "rescue", "rescued", "rescuing",
    "help", "assist", "assistance",
    "relief", "aid",
    "what to do", "how to",
    "steps", "procedure", "guidelines",
    "instructions", "plan", "preparedness",
    "safe", "safety", "precautions",
    "dos and donts",
    "shelter", "safe place",
    "first aid", "emergency kit",
    "move to", "avoid area",
    "seek help", "call emergency"
]

URGENCY_WORDS = [
    "now", "right now", "immediately",
    "urgent", "urgently", "asap",
    "alert", "warning", "red alert",
    "emergency", "emergency alert",
    "breaking", "just in", "live",
    "developing situation",
    "help needed", "need help",
    "people trapped", "casualties reported"
]

# -------------------------------
# MODEL LOADING
# -------------------------------

def load_intent_model():
    with open(INTENT_MODEL_PATH, "rb") as f:
        return pickle.load(f)

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------

def extract_intent_features(query: str):
    q = query.lower()

    disaster = int(any(w in q for w in DISASTER_WORDS))
    action = int(any(w in q for w in ACTION_WORDS))
    urgency = int(any(w in q for w in URGENCY_WORDS))

    return [[disaster, action, urgency]]

# -------------------------------
# INTENT INFERENCE
# -------------------------------

def is_emergency(query, model, threshold=0.6):
    X = extract_intent_features(query)
    prob = model.predict_proba(X)[0][1]
    return prob >= threshold, prob
