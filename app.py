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
    page_icon="🎧",
    layout="wide"
)

# =========================================================
# SAFE MOOD → KEYWORD MAP (NO KEYERROR POSSIBLE)
# =========================================================
MOOD_KEYWORDS = {
    "happy": ["fun", "uplifting", "comedy", "joy", "positive"],
    "sad": ["emotional", "healing", "soft", "calm", "comfort"],
    "focused": ["deep focus", "productivity", "study", "learning"],
    "relaxed": ["chill", "lofi", "peaceful", "ambient"],
    "adventurous": ["travel", "explore", "thrill", "experience"],
    "romantic": ["love", "romantic", "emotional", "bonding"]
}

DEFAULT_KEYWORDS = ["popular", "trending", "recommended"]

# =========================================================
# THEME TOGGLE
# =========================================================
dark_mode = st.toggle("🌗 Dark / Light Mode")

if dark_mode:
    st.markdown(
        """
        <style>
        body { background-color: #0f172a; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# TITLE
# =========================================================
st.markdown(
    "<h1 style='text-align:center;'>🎯 Mood-Aware Smart Recommendation System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>AI that understands mood, intent & vibe</p>",
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# LOAD DATA (FAST, LOCAL, SAFE)
# =========================================================
@st.cache_data
def load_data():
    data = {
        "title": [
            "Feel-Good Comedy Show",
            "Deep Focus Study Playlist",
            "Romantic Evening Music",
            "Chill Lo-Fi Beats",
            "Adventure Travel Documentary",
            "Motivational Success Talk",
            "Emotional Healing Podcast"
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
# NLP MODEL
# =========================================================
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["description"])

# =========================================================
# SESSION STATE
# =========================================================
if "mood" not in st.session_state:
    st.session_state.mood = "happy"

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("🎛 Personalization")

mood_label_map = {
    "😀 Happy": "happy",
    "😢 Sad": "sad",
    "🎯 Focused": "focused",
    "😌 Relaxed": "relaxed",
    "🚀 Adventurous": "adventurous",
    "❤️ Romantic": "romantic"
}

selected_label = st.sidebar.radio(
    "Select your mood",
    list(mood_label_map.keys())
)

st.session_state.mood = mood_label_map[selected_label]

intent = st.sidebar.text_input(
    "🎯 What are you looking for?",
    placeholder="music, podcast, learning, travel..."
)

chaos_mode = st.sidebar.toggle("🔥 Chaos Mode (Surprise Me)")

# =========================================================
# RECOMMENDATION FUNCTION (KEYERROR-PROOF)
# =========================================================
def recommend():
    mood = st.session_state.mood

    keywords = MOOD_KEYWORDS.get(mood, DEFAULT_KEYWORDS)

    if chaos_mode:
        keywords = random.choice(list(MOOD_KEYWORDS.values()))

    query = " ".join(keywords) + " " + intent
    query_vec = vectorizer.transform([query])

    scores = cosine_similarity(query_vec, X)[0]
    df["score"] = scores

    top_results = df.sort_values("score", ascending=False).head(3)
    confidence = round(max(scores) * 100, 2)

    return top_results, confidence, keywords

# =========================================================
# MAIN BUTTON
# =========================================================
if st.button("🚀 Recommend Now", use_container_width=True):
    results, confidence, used_keywords = recommend()

    st.success(f"✨ Confidence Score: {confidence}%")
    st.caption(f"🔑 AI Keywords Used: {', '.join(used_keywords)}")

    st.divider()

    for _, row in results.iterrows():
        st.markdown(
            f"""
            ### 🎬 {row['title']}
            📝 *{row['description']}*
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            st.button("👍 Like", key=f"like_{row['title']}")
        with col2:
            st.button("👎 Dislike", key=f"dislike_{row['title']}")

        st.divider()

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    "<p style='text-align:center; font-size:12px;'>Built with ❤️ by Lavisha | AI-Powered Recommendation System</p>",
    unsafe_allow_html=True
)
