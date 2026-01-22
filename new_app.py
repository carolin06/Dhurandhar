import streamlit as st
import json
import pickle
from datetime import datetime, timezone
import streamlit.components.v1 as components
import math
import time

# =========================
# CONFIG
# =========================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "documents_2000_upgradedloc.json"
INTENT_MODEL_PATH = BASE_DIR / "src" / "intent_model.pkl"
VERIFY_MODEL_PATH = BASE_DIR / "src" / "verification_model.pkl"

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
raw_docs = load_documents()

def save_docs(docs):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=4, ensure_ascii=False)


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

STOPWORDS = {
    "what", "is", "the", "a", "an", "how", "why",
    "when", "where", "who", "does", "do", "did"
}

IMMEDIATE_WORDS = {
    "here", "near", "nearby", "around", "local", "my area", "close"
}

#========================
#location detection

KNOWN_LOCATIONS = {
    # --- Major Metros ---
    "delhi": (28.6139, 77.2090),
    "new delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "bengaluru": (12.9716, 77.5946),
    "bangalore": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714),

    # --- Uttar Pradesh ---
    "prayagraj": (25.4358, 81.8463),
    "allahabad": (25.4358, 81.8463),
    "lucknow": (26.8467, 80.9462),
    "kanpur": (26.4499, 80.3319),
    "varanasi": (25.3176, 82.9739),
    "agra": (27.1767, 78.0081),
    "meerut": (28.9845, 77.7064),
    "ghaziabad": (28.6692, 77.4538),
    "noida": (28.5355, 77.3910),

    # --- Bihar ---
    "patna": (25.5941, 85.1376),
    "gaya": (24.7914, 85.0002),
    "bhagalpur": (25.2425, 86.9842),

    # --- West Bengal ---
    "howrah": (22.5958, 88.2636),
    "durgapur": (23.5204, 87.3119),
    "asansol": (23.6739, 86.9524),
    "siliguri": (26.7271, 88.3953),

    # --- Odisha (cyclone-prone) ---
    "bhubaneswar": (20.2961, 85.8245),
    "cuttack": (20.4625, 85.8830),
    "puri": (19.8135, 85.8312),
    "balasore": (21.4942, 86.9310),
    "berhampur": (19.3149, 84.7941),

    # --- Andhra Pradesh (cyclone-prone) ---
    "visakhapatnam": (17.6868, 83.2185),
    "vizag": (17.6868, 83.2185),
    "vijayawada": (16.5062, 80.6480),
    "nellore": (14.4426, 79.9865),
    "kakinada": (16.9891, 82.2475),

    # --- Tamil Nadu ---
    "coimbatore": (11.0168, 76.9558),
    "madurai": (9.9252, 78.1198),
    "tiruchirappalli": (10.7905, 78.7047),
    "trichy": (10.7905, 78.7047),
    "salem": (11.6643, 78.1460),
    "vellore": (12.9165, 79.1325),

    # --- Kerala ---
    "thiruvananthapuram": (8.5241, 76.9366),
    "trivandrum": (8.5241, 76.9366),
    "kochi": (9.9312, 76.2673),
    "ernakulam": (9.9816, 76.2999),
    "kozhikode": (11.2588, 75.7804),
    "calicut": (11.2588, 75.7804),

    # --- Assam / North-East ---
    "guwahati": (26.1445, 91.7362),
    "silchar": (24.8333, 92.7789),
    "jorhat": (26.7509, 94.2037),
    "dibrugarh": (27.4728, 94.9110),
    "imphal": (24.8170, 93.9368),
    "aizawl": (23.7271, 92.7176),
    "agartala": (23.8315, 91.2868),
    "shillong": (25.5788, 91.8933),
    "kohima": (25.6701, 94.1077),

    # --- Rajasthan ---
    "jaipur": (26.9124, 75.7873),
    "jodhpur": (26.2389, 73.0243),
    "udaipur": (24.5854, 73.7125),
    "kota": (25.2138, 75.8648),
    "bikaner": (28.0229, 73.3119),

    # --- Gujarat ---
    "surat": (21.1702, 72.8311),
    "rajkot": (22.3039, 70.8022),
    "bhavnagar": (21.7645, 72.1519),
    "jamnagar": (22.4707, 70.0577),
    "porbandar": (21.6417, 69.6293),

    # --- Maharashtra ---
    "nagpur": (21.1458, 79.0882),
    "nashik": (19.9975, 73.7898),
    "aurangabad": (19.8762, 75.3433),
    "kolhapur": (16.7050, 74.2433),
    "solapur": (17.6599, 75.9064),

    # --- Disaster keywords treated as regions ---
    "assam": (26.2006, 92.9376),
    "odisha": (20.9517, 85.0985),
    "kerala": (10.8505, 76.2711),
    "bihar": (25.0961, 85.3131),
    "west bengal": (22.9868, 87.8550),
    "andhra pradesh": (15.9129, 79.7400),
    "tamil nadu": (11.1271, 78.6569),
    "uttar pradesh": (26.8467, 80.9462)
}


def extract_query_location(query):
    q = query.lower()
    for loc in KNOWN_LOCATIONS:
        if loc in q:
            return loc, KNOWN_LOCATIONS[loc]
    return None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# =========================
# FRESHNESS SCORE
# =========================
def freshness_score(timestamp, emergency=False):
    try:
        doc_time = datetime.fromisoformat(timestamp)
    except:
        return 1.0

    if doc_time.tzinfo is None:
        doc_time = doc_time.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    hours_old = (now - doc_time).total_seconds() / 3600

    tau = 24 if emergency else 72   # 👈 emergency decays faster
    raw = math.exp(-hours_old / tau)

    return max(raw, 0.05)  # 👈 HARD FLOOR (CRITICAL)

#============================

#session state for pogo
    
if "click_time" not in st.session_state:
    st.session_state.click_time = {}

if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "results"




def score_document(doc, emergency, query_words, user_coords=None,near_me=False):
    trust = float(doc["trust"])
    freshness = float(doc["freshness"])
    pogo = doc.get("pogo", 0)

    pogo_penalty = math.exp(-pogo / 5)

    text = (doc["title"] + " " + doc["text"] + " " + doc.get("location", "")).lower()
    relevance = sum(1 for w in query_words if w in text)

    base = (0.2 * trust + 0.8 * freshness) if emergency else trust
    score= (base + 0.1 * relevance) * pogo_penalty

       
    if near_me and user_coords and isinstance(doc.get("lat"), (int, float)):
        dist = haversine(user_coords[0], user_coords[1], doc["lat"], doc["lon"])
        distance_score = math.exp(-dist / 50)  # closer = higher
        score = distance_score  # distance dominates

    return score



# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Emergency-Aware Search Engine", layout="wide")
st.title("🚨 Emergency-Aware Search Engine")



components.html("""
<script>
navigator.geolocation.getCurrentPosition(
  (pos) => {
    const lat = pos.coords.latitude;
    const lon = pos.coords.longitude;
    const url = new URL(window.location);
    url.searchParams.set("lat", lat);
    url.searchParams.set("lon", lon);
    window.location.href = url.toString();
  }
);
</script>

""", height=0)

params = st.query_params
user_lat = float(params.get("lat", [None])[0]) if "lat" in params else None
user_lon = float(params.get("lon", [None])[0]) if "lon" in params else None
user_coords = (user_lat, user_lon) if user_lat and user_lon else None

# --- Manual location override for testing ---
manual_city = st.selectbox(
    "Use manual location (optional)",
    ["Auto-detect", "Prayagraj", "Lucknow", "Delhi", "Kolkata", "Kerala"]
)

if manual_city != "Auto-detect":
    user_coords = KNOWN_LOCATIONS[manual_city.lower()]

# Show detected location for debugging
if user_coords:
    st.warning(f"📍 Using coordinates: lat={user_coords[0]:.4f}, lon={user_coords[1]:.4f}")
else:
    st.error("📍 Location not detected")

query = st.text_input("Search", placeholder="Type your query and press Enter")
results = []



if query:
    # ---------- INTENT MODEL ----------
    
    X_intent = extract_intent_features(query)
    ml_emergency = int(intent_model.predict(X_intent)[0])

    q_lower = query.lower()
    near_me = any(
        w in query.lower()
        for w in ["near me", "nearby", "near", "around", "close"]
    )


    query_loc, query_coords = extract_query_location(query)

    if near_me and user_coords:
        effective_user_coords = user_coords
    elif query_coords:
        effective_user_coords = query_coords
    else:
        effective_user_coords = None




    rule_emergency = (
        any(w in q_lower for w in [
            "earthquake", "flood", "fire", "explosion",
            "cyclone", "tsunami", "landslide"
        ])
        and any(w in q_lower for w in IMMEDIATE_WORDS)
    )

    emergency = int(ml_emergency or rule_emergency)

    urgency = X_intent[0][2]

    if emergency:
        st.error("🚨 Emergency Mode ON (ML-Detected)")
    else:
        st.success("✅ Normal Mode")

    
    query_words = [
    w for w in query.lower().split()
    if w not in STOPWORDS
    and w not in {"near", "nearby", "around", "me", "local"}
    ]



    for doc in docs:
        text = (doc["title"] + " " + doc["text"]+ doc["location"]).lower() 
        match_count = sum(1 for w in query_words if w in text)
        MAX_DISTANCE_KM = 300   # adjust as needed

        if effective_user_coords and isinstance(doc.get("lat"), (int, float)):
            dist = haversine(
                effective_user_coords[0], effective_user_coords[1],
                doc["lat"], doc["lon"]
            )
            if dist > MAX_DISTANCE_KM:
                continue
            distance_score = math.exp(-dist / 50)
        else:
                distance_score = 0

        if match_count >= 1:

            # ---------- VERIFICATION MODEL ----------
            X_verify = extract_verification_features(doc, urgency)
            trust_prob = verify_model.predict_proba(X_verify)[0][1]

            freshness = freshness_score(
                doc.get("timestamp", ""),
                emergency=emergency
            )
            #if emergency and freshness < 0.3:
                #continue


            result = {
                **doc,
                "trust": trust_prob,
                "freshness": freshness,
                "_score": score_document(
                    {**doc, "trust": trust_prob, "freshness": freshness},
                    emergency,
                    query_words,
                    user_coords=effective_user_coords,
                    near_me=near_me
                )


            }

            results.append(result)

    results.sort(key=lambda x: x["_score"], reverse=True)
    RESULTS_PER_PAGE = 10
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=max(1, math.ceil(len(results) / RESULTS_PER_PAGE)),
        step=1
    )

    start = (page - 1) * RESULTS_PER_PAGE
    end = start + RESULTS_PER_PAGE

    if st.session_state.view_mode == "results" :
        st.write(f"Found {len(results)} result(s)")


        for i, doc in enumerate(results[start:end], start=start + 1):
            
            st.markdown(f"### {i}. {doc['title']}")
            if st.button("Open", key=f"open_{doc['id']}"):

                st.session_state.click_time[doc["id"]] = time.time()
                st.session_state.current_doc_id = doc["id"]
                st.session_state.view_mode = "doc"
                st.rerun()
            
            st.caption(
                f"Trust: {doc['trust']:.2f} | "
                f"Freshness: {doc['freshness']:.2f} | "
                f"Score: {doc['_score']:.3f} | "
                f"Time: {doc.get('timestamp','')}"
            )

            st.divider()

if st.session_state.view_mode == "doc":
    doc_id = st.session_state.current_doc_id
    doc = next(d for d in docs if d["id"] == doc_id)

    st.subheader(doc["title"])
    st.write(doc["text"])

    if st.button("← Back to results"):
        POGO_THRESHOLD = 8  # seconds
        end_time = time.time()

        if doc_id in st.session_state.click_time:
            start_time = st.session_state.click_time[doc_id]
            dwell_time = end_time - start_time
        else:
            dwell_time = POGO_THRESHOLD + 1  # no pogo

        if dwell_time < POGO_THRESHOLD:
            doc["pogo"] = doc.get("pogo", 0) + 1
            
            save_docs(docs)
            st.warning("Quick return detected (pogo-sticking)")
        else:
            st.success("User engaged — no pogo")

        st.session_state.current_doc_id = None
        st.session_state.view_mode = "results"
        st.rerun()




