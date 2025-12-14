import streamlit as st
import pandas as pd
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mood_map import mood_keywords

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="NeuroPulse AI", page_icon="🧠", layout="wide")

# --------------------------------------------------
# THEME TOGGLE
# --------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def switch_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# --------------------------------------------------
# STYLES
# --------------------------------------------------
def load_css(theme):
    if theme == "dark":
        bg = "#0e1117"
        card = "#161b22"
        text = "#fafafa"
        sub = "#9ca3af"
    else:
        bg = "#f9fafb"
        card = "#ffffff"
        text = "#111827"
        sub = "#6b7280"

    st.markdown(f"""
    <style>
    body {{ background-color: {bg}; color: {text}; }}
    .header {{ font-size: 42px; font-weight: 700; }}
    .sub {{ font-size: 16px; color: {sub}; }}
    .card {{
        background: {card};
        padding: 20px;
        border-radius: 14px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}
    </style>
    """, unsafe_allow_html=True)

load_css(st.session_state.theme)

# --------------------------------------------------
# DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()
tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(df["description"])

# --------------------------------------------------
# HEADER
# --------------------------------------------------
top_left, top_right = st.columns([4,1])

with top_left:
    st.markdown("<div class='header'>🧠 NeuroPulse</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Emotion-aware AI recommendation system</div>", unsafe_allow_html=True)

with top_right:
    st.toggle("🌗 Dark / Light", on_change=switch_theme)

st.markdown("---")

# --------------------------------------------------
# EMOJI MOOD SELECTOR
# --------------------------------------------------
mood_emojis = {
    "Happy": "😄",
    "Sad": "😔",
    "Focused": "🎯",
    "Relaxed": "🌿",
    "Adventurous": "🚀",
    "Romantic": "❤️"
}

if "selected_mood" not in st.session_state:
    st.session_state.selected_mood = "Happy"

st.subheader("😊 Select Your Mood")

cols = st.columns(len(mood_emojis))
for col, (mood, emoji) in zip(cols, mood_emojis.items()):
    if col.button(f"{emoji}\n{mood}"):
        st.session_state.selected_mood = mood

st.success(f"Selected Mood: {st.session_state.selected_mood}")

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
intent = st.text_input("🔎 What are you in the mood for?")
energy = st.slider("⚡ Energy Level", 1, 10, 5)
chaos = st.toggle("😈 Chaos Mode")

# --------------------------------------------------
# RECOMMENDATION LOGIC
# --------------------------------------------------
def recommend():
    mood = st.session_state.selected_mood
    keywords = mood_keywords[mood]

    if chaos:
        keywords = random.choice(list(mood_keywords.values()))

    query = " ".join(keywords) + " " + intent
    q_vec = tfidf.transform([query])
    scores = cosine_similarity(q_vec, X)[0]

    df["score"] = scores
    top = df.sort_values("score", ascending=False).head(3)
    return top, max(scores), keywords

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
if st.button("🚀 Generate Recommendations", use_container_width=True):
    with st.spinner("Thinking like your brain 🧠..."):
        time.sleep(1)

    results, confidence, keys = recommend()

    st.subheader("🎁 Recommended For You")

    for i, r in results.iterrows():
        st.markdown(f"""
        <div class='card'>
            <b>{r['title']}</b><br>
            <small>{r['type']}</small><br><br>
            {r['description']}
        </div>
        """, unsafe_allow_html=True)

        # Feedback
        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            st.button("👍 Helpful", key=f"up_{i}")
        with fb_col2:
            st.button("👎 Not Helpful", key=f"down_{i}")

    st.progress(min(int(confidence * 100), 100))
    st.info(f"AI Confidence: {int(confidence*100)}% | Keywords used: {keys}")

    if chaos:
        st.warning("Chaos Mode ON — unexpected but interesting picks 😈")
