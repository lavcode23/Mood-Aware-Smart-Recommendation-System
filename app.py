import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mood_map import mood_keywords

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="Emotion AI Recommender",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CSS MAGIC ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
}
.glass {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 15px;
}
.big {
    font-size: 26px;
    font-weight: bold;
}
.explain {
    font-size: 14px;
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# ---------------- VECTORIZATION ----------------
tfidf = TfidfVectorizer(stop_words="english")
content_matrix = tfidf.fit_transform(df["description"])

# ---------------- TITLE ----------------
st.markdown("<div class='big'>🧠 Emotion-Driven AI Recommendation Engine</div>", unsafe_allow_html=True)
st.write("Not what you click. **How you FEEL.**")

# ---------------- USER INPUT ----------------
mood = st.selectbox("✨ Select your current mood", list(mood_keywords.keys()))
intent = st.text_input("💬 What do you want right now?", placeholder="relax, fun, focus, surprise")
energy = st.slider("⚡ Energy Level", 1, 10, 5)

surprise_mode = st.toggle("🎁 Surprise Me")

# ---------------- RECOMMENDATION LOGIC ----------------
def recommend(mood, intent, energy):
    mood_words = mood_keywords[mood]

    if intent.strip() == "":
        intent = random.choice(mood_words)

    query = " ".join(mood_words) + " " + intent
    query_vec = tfidf.transform([query])

    similarity = cosine_similarity(query_vec, content_matrix)[0]
    df["score"] = similarity

    results = df.sort_values("score", ascending=False)

    # Diversity filter
    seen = set()
    final = []
    for _, row in results.iterrows():
        if row["type"] not in seen:
            final.append(row)
            seen.add(row["type"])
        if len(final) == 3:
            break

    return final, mood_words

# ---------------- RUN ----------------
if st.button("🚀 Generate My Recommendations"):
    results, mood_words = recommend(mood, intent, energy)

    st.subheader("🎯 Your AI Picks")
    for r in results:
        st.markdown(
            f"<div class='glass'><b>{r['title']}</b><br>"
            f"<i>{r['type']}</i><br>"
            f"<div class='explain'>{r['description']}</div></div>",
            unsafe_allow_html=True
        )

    st.success(
        f"🧠 AI Reasoning: You feel **{mood}**, energy level **{energy}**, "
        f"and your intent matched keywords → {', '.join(mood_words)}"
    )

    if surprise_mode:
        st.balloons()
