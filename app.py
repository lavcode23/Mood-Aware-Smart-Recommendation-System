import streamlit as st
import pandas as pd
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mood_map import mood_keywords

st.set_page_config(page_title="NeuroPulse AI", page_icon="🧠")

# ---------------- CSS ----------------
st.markdown("""
<style>
body { background: linear-gradient(135deg,#0f0c29,#302b63,#24243e); color:white; }
.card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 15px;
}
.confidence { font-size:14px; opacity:0.85; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(df["description"])

# ---------------- UI ----------------
st.title("🧠 NeuroPulse")
st.caption("Emotion-aware AI recommendations. Not clicks. Feelings.")

mood = st.selectbox("Mood", list(mood_keywords.keys()))
intent = st.text_input("What do you want right now?")
energy = st.slider("Energy Level", 1, 10, 5)

chaos = st.toggle("😈 Chaos Mode")

# ---------------- RECOMMEND ----------------
def recommend():
    keywords = mood_keywords[mood]

    if chaos:
        keywords = random.choice(list(mood_keywords.values()))

    if intent.strip() == "":
        intent_used = random.choice(keywords)
    else:
        intent_used = intent

    query = " ".join(keywords) + " " + intent_used
    q_vec = tfidf.transform([query])
    scores = cosine_similarity(q_vec, X)[0]

    df["score"] = scores
    ranked = df.sort_values("score", ascending=False)

    seen = set()
    final = []
    for _, row in ranked.iterrows():
        if row["type"] not in seen:
            final.append(row)
            seen.add(row["type"])
        if len(final) == 3:
            break

    return final, max(scores), keywords

# ---------------- RUN ----------------
if st.button("🚀 Generate"):
    with st.spinner("🧠 Reading your vibe..."):
        time.sleep(1.5)

    results, confidence, keys = recommend()

    for r in results:
        st.markdown(
            f"<div class='card'><b>{r['title']}</b><br>"
            f"{r['type']}<br>{r['description']}</div>",
            unsafe_allow_html=True
        )

    st.progress(min(int(confidence * 100), 100))
    st.write(f"AI Confidence: {int(confidence*100)}%")
    st.success(f"Reason: mood={mood}, energy={energy}, matched keywords={keys}")

    if chaos:
        st.balloons()
