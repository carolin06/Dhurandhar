import streamlit as st
import json
import os
from datetime import datetime, timezone
import math
import re

# ==============================
# CONFIG
# ==============================
DATA_DIR = "data"
DOCUMENT_FILE = os.path.join(DATA_DIR, "documents_2000.json")

EMERGENCY_WORDS = {
    "earthquake", "flood", "cyclone", "tsunami",
    "explosion", "fire", "landslide", "storm",
    "evacuation", "alert", "emergency", "rescue"
}

# ==============================
# SAFE DOCUMENT LOADER
# (Fixes blank page issues)
# ==============================
def load_documents():
    try:
        if not os.path.exists(DOCUMENT_FILE):
            st.error(f"❌ File not found: {DOCUMENT_FILE}")
            st.stop()

        with open(DOCUMENT_FILE, "r", encoding="utf-8") as f:
            docs = json.load(f)

        if not isinstance(docs, list):
            st.error("❌ Document file format is invalid")
            st.stop()

        return docs

    except Exception as e:
        st.error("❌ Failed to load documents")
        st.exception(e)
        st.stop()

# ==============================
# UTILS
# ==============================
def tokenize(text):
    return set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))

def is_emergency_query(query):
    q_words = tokenize(query)
    return any(w in EMERGENCY_WORDS for w in q_words)

# ==============================
# FRESHNESS SCORE (SAFE)
# ==============================
def freshness_score(timestamp):
    try:
        doc_time = datetime.fromisoformat(str(timestamp))

        if doc_time.tzinfo is None:
            doc_time = doc_time.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        hours_old = (now - doc_time).total_seconds() / 3600

        return 1 / (1 + hours_old)

    except:
        return 0.5

# ==============================
# RELEVANCE SCORE
# ==============================
def relevance_score(query_words, doc_text):
    doc_words = tokenize(doc_text)
    if not doc_words:
        return 0.0
    return len(query_words & doc_words) / len(query_words)

# ==============================
# FINAL SCORING LOGIC
# ==============================
def score_document(doc, query_words, emergency):
    trust = float(doc.get("trust", 0.5))
    text = f"{doc.get('title','')} {doc.get('text','')}"
    relevance = relevance_score(query_words, text)
    fresh = freshness_score(doc.get("timestamp"))

    if emergency:
        # EMERGENCY MODE → Trust + Freshness dominate
        score = (
            0.45 * trust +
            0.35 * fresh +
            0.20 * relevance
        )
    else:
        # NORMAL MODE → Relevance dominates
        score = (
            0.60 * relevance +
            0.25 * trust +
            0.15 * fresh
        )

    return score

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(
    page_title="Emergency-Aware Search Engine",
    layout="wide"
)

st.title("🚨 Emergency-Aware Search Engine")

# Load docs safely
documents = load_documents()

query = st.text_input("Search", placeholder="e.g. flood alert in prayagraj")

if query:
    query_words = tokenize(query)
    emergency = is_emergency_query(query)

    if emergency:
        st.error("🚨 Emergency Mode ON (Trusted + Latest)")
    else:
        st.success("✅ Normal Mode (Most Relevant)")

    results = []

    for doc in documents:
        text = f"{doc.get('title','')} {doc.get('text','')}".lower()
        if any(w in text for w in query_words):
            doc["_score"] = score_document(doc, query_words, emergency)
            results.append(doc)

    results = sorted(results, key=lambda d: d["_score"], reverse=True)

    st.write(f"### Found {len(results)} result(s)")

    for i, d in enumerate(results[:10], start=1):
        st.markdown(f"### {i}. {d.get('title','(No title)')}")
        st.write(d.get("text","")[:300] + "...")
        st.caption(
            f"Trust: {d.get('trust',0.5)} | "
            f"Score: {round(d['_score'],3)} | "
            f"Time: {d.get('timestamp','')}"
        )
        st.divider()

