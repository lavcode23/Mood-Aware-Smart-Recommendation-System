import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Mood-Aware Smart Recommendation System",
    page_icon="💥",
    layout="wide"
)

# =========================================================
# CUSTOM INSANE UI STYLES
# =========================================================
st.markdown("""
<style>
body {
    background: linear-gradient(-45deg, #0f172a, #020617, #020617, #0f172a);
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 0 40px rgba(0,0,0,0.4);
}

.title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #cbd5f5;
    margin-bottom: 25px;
}

.confidence {
    font-size: 1.2rem;
    color: #38bdf8;
    font-weight: bold;
}

button[kind="primary"] {
    border-radius: 14px;
    height: 3.2em;
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="title">💥 Mood-Aware Smart Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI that understands your mood, intent & vibe</div>', unsafe_allow_html=True)

# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_data():
    data = {
        "title": [
            "😂 Feel-Good Comedy Show",
            "🎯 Deep Focus Study Playlist",
            "❤️ Romantic Evening Music",
            "😌 Chill Lo-Fi Beats",
            "🚀 Adventure Travel Documentary",
            "🔥 Motivational Success Talk",
            "🩹 Emotional Healing Podcast"
        ],
        "description": [
            "fun uplifting comedy entertainment",
            "deep focus productivity study learning",
            "romantic love emotional bonding",
            "calm peaceful ambient chill music",
            "travel explore adventure thrill experience",
            "motivation success growth mindset",
            "emotional comfort healing calm podcast"
        ]
    }
    return pd.DataFrame(data)

df = load_data()

# =========================================================
# NLP
# =========================================================
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["description"])

MOOD_MAP = {
    "😀 Happy": ["fun", "uplifting", "positive"],
    "😢 Sad": ["emotional", "healing", "comfort"],
    "🎯 Focused": ["study", "learning", "productivity"],
    "😌 Relaxed": ["chill", "ambient", "peaceful"],
    "🚀 Adventurous": ["travel", "explore", "thrill"],
    "❤️ Romantic": ["love", "romantic", "bonding"]
}

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("🎛 Personalization")

mood_label = st.sidebar.radio("Select Mood", list(MOOD_MAP.keys()))
intent = st.sidebar.text_input("🎯 What do you want?", placeholder="music, learning, podcast...")
chaos_mode = st.sidebar.toggle("🔥 Chaos Mode")

# =========================================================
# RECOMMENDER
# =========================================================
def recommend():
    keywords = MOOD_MAP[mood_label]
    if chaos_mode:
        keywords = random.choice(list(MOOD_MAP.values()))

    query = " ".join(keywords) + " " + intent
    q_vec = vectorizer.transform([query])

    scores = cosine_similarity(q_vec, X)[0]
    df["score"] = scores

    return df.sort_values("score", ascending=False).head(3), round(max(scores)*100, 2), keywords

# =========================================================
# ACTION
# =========================================================
if st.button("🚀 Generate Recommendations", use_container_width=True):
    results, confidence, used_keywords = recommend()

    st.markdown(f'<div class="confidence">✨ Confidence Score: {confidence}%</div>', unsafe_allow_html=True)
    st.caption("🔑 Keywords used: " + ", ".join(used_keywords))

    for _, row in results.iterrows():
        st.markdown(f"""
        <div class="glass">
            <h3>{row['title']}</h3>
            <p>{row['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.button("👍 Loved it", key=f"like_{row['title']}")
        with col2:
            st.button("👎 Not my vibe", key=f"dislike_{row['title']}")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    "<p style='text-align:center; opacity:0.6;'>Built with ❤️ by Lavisha • AI Resume Project</p>",
    unsafe_allow_html=True
)
