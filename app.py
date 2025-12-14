import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mood_map import mood_keywords

st.set_page_config(page_title="Mood Recommendation", page_icon="🧠")

st.title("🧠 Mood-Aware Recommendation System")
st.write("Recommendations based on how you feel ✨")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# Mood input
mood = st.selectbox("How are you feeling right now?", list(mood_keywords.keys()))
intent = st.text_input("What do you want right now? (relax, fun, focus...)")

# Vectorize content
tfidf = TfidfVectorizer(stop_words="english")
content_matrix = tfidf.fit_transform(df["description"])

def recommend(mood, intent):
    mood_words = mood_keywords[mood]
    query = " ".join(mood_words) + " " + intent

    query_vec = tfidf.transform([query])
    similarity = cosine_similarity(query_vec, content_matrix)[0]

    df["score"] = similarity
    results = df.sort_values("score", ascending=False).head(3)

    return results, mood_words

if st.button("✨ Recommend for me"):
    results, mood_words = recommend(mood, intent)

    st.subheader("🔮 Recommendations for you:")
    for _, row in results.iterrows():
        st.success(f"{row['title']} ({row['type']})")

    st.info(f"💡 Because you feel **{mood}** and mentioned **{intent}**, we matched content related to: {', '.join(mood_words)}")
