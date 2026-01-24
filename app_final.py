import streamlit as st
import json
import pickle
from datetime import datetime, timezone
import math
import time
import sys
import pandas as pd
# =========================
# CONFIG
# =========================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data"/"documents_2000_upgraded_st.json"
INTENT_MODEL_PATH = BASE_DIR / "models" / "intent_model.pkl"
VERIFY_MODEL_PATH = BASE_DIR / "models" / "verification_model.pkl"

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




def score_document(doc, emergency):
    trust = float(doc["trust"])
    freshness = float(doc["freshness"])
    pogo = doc.get("pogo", 0)

    pogo_penalty = math.exp(-pogo / 5)

    text = (doc["title"] + " " + doc["text"] + " " + doc.get("location", "")).lower()
    relevance = sum(1 for w in query_words if w in text)

    base = (0.6 * trust + 0.4 * freshness) if emergency else trust
    return (base + 0.1 * relevance) * pogo_penalty

# =========================
# ADMIN PANEL CONFIG
# =========================
def admin_login():
    st.subheader("🔐 Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.admin_logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")


# =========================
# STREAMLIT UI
# =========================

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Search Engine", "Admin Panel"]
)

if mode == "Search Engine":
    
    st.set_page_config(page_title="Emergency-Aware Search Engine", layout="wide")
    st.title("🚨 Emergency-Aware Search Engine")

    query = st.text_input("Search", placeholder="Type your query and press Enter")
    results = []

    #autocorrect

    sys.path.append(str(BASE_DIR / "src"))

    from autocorrect import autocorrect_query

    if query:
        # ---------- INTENT MODEL ----------
        X_intent = extract_intent_features(query)
        ml_emergency = int(intent_model.predict(X_intent)[0])

        q = autocorrect_query(query)
        if q != query:
            st.info(f"Did you mean: **{q}** ?")

        query = q
        q_lower = query.lower()
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
        ]


        for doc in docs:
            text = (doc["title"] + " " + doc["text"]).lower()
            match_count = sum(1 for w in query_words if w in text)

            if match_count >= 1:

                # ---------- VERIFICATION MODEL ----------
                X_verify = extract_verification_features(doc, urgency)
                trust_prob = verify_model.predict_proba(X_verify)[0][1]

                freshness = freshness_score(
                    doc.get("timestamp", ""),
                    emergency=emergency
                )

                result = {
                    **doc,
                    "trust": trust_prob,
                    "freshness": freshness,
                    "_score": score_document(
                        {**doc, "trust": trust_prob, "freshness": freshness},
                        emergency
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
            POGO_THRESHOLD = 15  # seconds
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

elif mode == "Admin Panel":
    if not st.session_state.admin_logged_in:
        admin_login()
        
    else:
        col1, col2 = st.columns([6,1])
        with col2:
            if st.button("Logout"):
                st.session_state.admin_logged_in = False
                st.rerun()

        st.title("🛠 Admin Panel")

        # ---------- TABS ----------
        tab1, tab2 = st.tabs(["📂 Document Database", "➕ Add Update"])

        with tab1:
            # ---------- DATABASE VIEW ----------
            docs = load_documents()
            df_docs = pd.DataFrame(docs)
            df_docs["timestamp"] = pd.to_datetime(df_docs["timestamp"], errors="coerce")

            st.subheader("📂 Document Database (Read-only)")

            # SIDEBAR FILTERS
            st.sidebar.markdown("### 🔎 Admin Filters")

            selected_cols = st.sidebar.multiselect(
                "Columns",
                df_docs.columns.tolist(),
                default=["title", "source_type", "timestamp"]
            )

            search_title = st.sidebar.text_input("Search title")
            source_filter = st.sidebar.multiselect(
                "Source type",
                df_docs["source_type"].unique()
            )

            emergency_only = st.sidebar.checkbox("Emergency only")

            min_d, max_d = df_docs["timestamp"].min(), df_docs["timestamp"].max()
            date_range = st.sidebar.date_input(
                "Date range",
                value=(min_d.date(), max_d.date())
            )

            page_size = st.sidebar.selectbox("Rows per page", [10, 25, 50, 100], index=1)

            # APPLY FILTERS
            df = df_docs.copy()

            if search_title:
                df = df[df["title"].str.contains(search_title, case=False, na=False)]

            if source_filter:
                df = df[df["source_type"].isin(source_filter)]

            if emergency_only:
                pattern = "|".join(EMERGENCY_TOPICS)
                df = df[df["text"].str.lower().str.contains(pattern, na=False)]

            if date_range:
                if isinstance(date_range, (tuple, list)):
                    start_date, end_date = date_range
                else:
                    start_date = end_date = date_range

                start_dt = pd.to_datetime(start_date).tz_localize("UTC")
                end_dt = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=1)

                df = df[
                    (df["timestamp"] >= start_dt) &
                    (df["timestamp"] < end_dt)
                ]

            df = df.sort_values("timestamp", ascending=False)
            df = df[selected_cols]

            # PAGINATION
            total_pages = max(1, (len(df) - 1) // page_size + 1)
            page = st.number_input("Page", 1, total_pages)

            start = (page - 1) * page_size
            end = start + page_size

            st.dataframe(df.iloc[start:end], use_container_width=True)
            st.caption(f"Showing {start+1}–{min(end, len(df))} of {len(df)}")

        with tab2:  
            title = st.text_input("Title")
            content = st.text_area("Content")
            source_type = st.selectbox("Source Type", ["official", "social", "unknown"])

            if st.button("Add Update"):
                if title and content:
                    new_doc = {
                        "id": int(time.time() * 1000),
                        "title": title,
                        "text": content,
                        "source_type": source_type,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trust": 0.9 if source_type == "Official" else 0.6,
                        "pogo": 0
                    }
                    raw_docs = load_documents()
                    raw_docs.append(new_doc)
                    save_docs(raw_docs)

                    docs[:] = load_documents()  # refresh in-memory

                    st.success("✅ Update added")
                    st.rerun()
                else:
                    st.warning("Fill all fields")


