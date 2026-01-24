import streamlit as st
import json
import pickle
from datetime import datetime, timezone
import math

# =========================
# CONFIG
# =========================
DATA_FILE = "data/documents_2000.json"
INTENT_MODEL_PATH = "models/intent_model.pkl"
VERIFY_MODEL_PATH = "models/verification_model.pkl"

# =========================
# LOAD MODELS
# =========================
with open(INTENT_MODEL_PATH, "rb") as f:
    intent_model = pickle.load(f)

with open(VERIFY_MODEL_PATH, "rb") as f:
    verify_model = pickle.load(f)

# =========================
# LOAD DOCUMENTS
# =========================
def load_documents():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

docs = load_documents()

# =========================
# FEATURE EXTRACTION
# =========================
def extract_intent_features(query):
    q = query.lower()

    disaster = int(any(w in q for w in [
        "earthquake", "flood", "cyclone", "tsunami",
        "fire", "explosion", "landslide", "disaster"
    ]))

    action = int(any(w in q for w in [
        "evacuate", "rescue", "help", "alert",
        "shelter", "warning", "emergency"
    ]))

    urgency = int(any(w in q for w in [
        "urgent", "now", "immediately", "today",
        "asap", "danger"
    ]))

    return [[disaster, action, urgency]]


def extract_verification_features(doc, urgency):
    trust = float(doc.get("trust", 0.5))
    gov_source = int(doc.get("source_type", "").lower() == "official")
    panic = int(any(w in doc.get("text", "").lower()
                    for w in ["panic", "fear", "chaos", "terror"]))

    return [[trust, gov_source, panic, urgency]]

# =========================
# FRESHNESS SCORE
# =========================
def freshness_score(timestamp):
    try:
        doc_time = datetime.fromisoformat(timestamp)
    except:
        return 1.0

    if doc_time.tzinfo is None:
        doc_time = doc_time.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    hours_old = (now - doc_time).total_seconds() / 3600

    return math.exp(-hours_old / 48)

# =========================
# FINAL RANKING
# =========================
def score_document(doc, emergency, urgency):
    trust = float(doc.get("trust", 0.5))
    freshness = freshness_score(doc.get("timestamp", ""))

    if emergency:
        return 0.6 * trust + 0.4 * freshness
    else:
        return trust

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Emergency-Aware Search Engine", layout="wide")
st.title("🚨 Emergency-Aware Search Engine")

query = st.text_input("Search", placeholder="Type your query and press Enter")

if query:
    # ---------- INTENT MODEL ----------
    X_intent = extract_intent_features(query)
    emergency = int(intent_model.predict(X_intent)[0])
    urgency = X_intent[0][2]

    if emergency:
        st.error("🚨 Emergency Mode ON (ML-Detected)")
    else:
        st.success("✅ Normal Mode")

    results = []

    for doc in docs:
        text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
        if query.lower() in text:
            # ---------- VERIFICATION MODEL ----------
            X_verify = extract_verification_features(doc, urgency)
            trust_prob = verify_model.predict_proba(X_verify)[0][1]

            doc["trust"] = trust_prob
            doc["_score"] = score_document(doc, emergency, urgency)
            results.append(doc)

    results.sort(key=lambda x: x["_score"], reverse=True)

    st.write(f"Found {len(results)} result(s)")

    for i, d in enumerate(results[:10], start=1):
        st.markdown(f"### {i}. {d.get('title','(No title)')}")
        st.write(d.get("text", "")[:250] + "...")
        st.caption(
            f"Trust: {round(d['trust'],2)} | "
            f"Score: {round(d['_score'],3)} | "
            f"Time: {d.get('timestamp','')}"
        )
        st.divider()


