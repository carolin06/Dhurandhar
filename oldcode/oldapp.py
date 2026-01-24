import streamlit as st
import json
import math
import time
from datetime import datetime, timezone
import os

# =========================
# CONFIG
# =========================
DATA_FILE = "documents_2000.json"

EMERGENCY_KEYWORDS = {
    "earthquake", "flood", "cyclone", "tsunami", "fire", "blast",
    "explosion", "evacuation", "landslide", "storm", "emergency",
    "alert", "rescue", "disaster"
}

# =========================
# UTILS
# =========================

def load_documents():
    if not os.path.exists(DATA_FILE):
        st.error(f"Document file not found: {DATA_FILE}")
        st.stop()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_timestamp(ts):
    """
    Always returns timezone-aware UTC datetime or None
    """
    if ts is None or ts == "":
        return None

    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)

        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
    except Exception:
        return None

    return None


def freshness_score(timestamp, emergency=False):
    """
    Exponential freshness decay.
    Faster decay during emergency.
    """
    doc_time = parse_timestamp(timestamp)
    if doc_time is None:
        return 1.0

    now = datetime.now(timezone.utc)
    hours_old = (now - doc_time).total_seconds() / 3600

    decay = 0.12 if emergency else 0.02
    return math.exp(-decay * hours_old)


def relevance_score(text, query_words):
    matches = sum(1 for w in query_words if w in text)
    return 1 + 0.3 * matches


def score_doc(doc, query_words, emergency=False):
    score = 1.0

    text = (doc.get("title", "") + " " + doc.get("text", "")).lower()

    # Relevance
    score *= relevance_score(text, query_words)

    # Trust
    score *= float(doc.get("trust", 0.5))

    # Freshness
    score *= freshness_score(doc.get("timestamp"), emergency)

    # Pogo penalty
    pogo = int(doc.get("pogo", 0))
    score *= max(0.3, 1 - 0.05 * pogo)

    return score


# =========================
# APP UI
# =========================

st.set_page_config(page_title="Emergency-Aware Search Engine", layout="wide")
st.title("🚨 Emergency-Aware Search Engine")

docs = load_documents()

query = st.text_input("Search", placeholder="e.g. flood in prayagraj")

if query:
    query_words = set(query.lower().split())
    emergency = any(w in EMERGENCY_KEYWORDS for w in query_words)

    if emergency:
        st.error("🚨 Emergency Mode ON (Trusted + Latest)")
    else:
        st.success("✅ Normal Mode (Most Relevant)")

    results = []

    for d in docs:
        text = (d.get("title", "") + " " + d.get("text", "")).lower()
        if any(w in text for w in query_words):
            d["_score"] = score_doc(d, query_words, emergency)
            results.append(d)

    results = sorted(results, key=lambda x: x["_score"], reverse=True)

    st.write(f"Found {len(results)} result(s)")

    for i, d in enumerate(results[:10], start=1):
        st.markdown(f"### {i}. {d.get('title', 'No title')}")
        st.write(d.get("text", "")[:300] + "...")
        st.caption(
            f"Trust: {d.get('trust', 0.5)} | "
            f"Pogo: {d.get('pogo', 0)} | "
            f"Time: {d.get('timestamp', 'unknown')}"
        )
        st.divider()
