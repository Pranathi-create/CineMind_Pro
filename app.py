# redeploy trigger
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import requests
import re

# ========================= 
# CONFIG
# =========================
TMDB_API_KEY = "eba858c4b0e9654aeacd93b1ce86c33e"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="CineMind Pro",
    page_icon="🎬",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    movies["genre_list"] = movies["genres"].str.split("|")
    return ratings, movies

ratings, movies = load_data()

# =========================
# ENCODING
# =========================
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings["userId_enc"] = user_encoder.fit_transform(ratings["userId"])
ratings["movieId_enc"] = movie_encoder.fit_transform(ratings["movieId"])

num_users = ratings["userId_enc"].nunique()
num_movies = ratings["movieId_enc"].nunique()

movie_id_to_title = dict(zip(movies["movieId"], movies["title"]))

# =========================
# MODEL (Inference Only)
# =========================
embedding_size = 50

user_input = tf.keras.layers.Input(shape=(1,))
movie_input = tf.keras.layers.Input(shape=(1,))

u = tf.keras.layers.Embedding(num_users, embedding_size)(user_input)
m = tf.keras.layers.Embedding(num_movies, embedding_size)(movie_input)

u = tf.keras.layers.Flatten()(u)
m = tf.keras.layers.Flatten()(m)

x = tf.keras.layers.Concatenate()([u, m])
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.models.Model([user_input, movie_input], output)
model.compile(optimizer="adam", loss="mse")

# =========================
# HELPERS
# =========================
def extract_year(title):
    match = re.search(r"\((\d{4})\)", title)
    return int(match.group(1)) if match else 0

def clean_title(title):
    return re.sub(r"\(\d{4}\)", "", title).strip()

@st.cache_data(show_spinner=False)
def get_movie_details(title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": clean_title(title),
        "language": "en-US"
    }
    r = requests.get(url, params=params).json()

    if r.get("results"):
        m = r["results"][0]
        poster_path = m.get("poster_path")
        poster = (
            TMDB_IMAGE_BASE + poster_path
            if poster_path else "https://via.placeholder.com/300x450?text=No+Poster"
        )
        return {
            "poster": poster,
            "overview": m.get("overview", "No overview available."),
            "release": m.get("release_date", "N/A")[:4]
        }

    return {
        "poster": "https://via.placeholder.com/300x450?text=No+Poster",
        "overview": "No overview available.",
        "release": "N/A"
    }

def recommend_movies(user_id, n=12):
    user_enc = user_encoder.transform([user_id])[0]
    movie_encs = np.arange(num_movies)
    user_array = np.full(num_movies, user_enc)

    preds = model.predict([user_array, movie_encs], verbose=0).flatten()
    top_idx = preds.argsort()[::-1]

    results = []
    for i in top_idx:
        movie_id = movie_encoder.inverse_transform([i])[0]
        results.append((movie_id_to_title[movie_id], float(preds[i])))
        if len(results) >= n * 2:
            break
    return results

def get_because_you_liked(user_id):
    top = (
        ratings[ratings["userId"] == user_id]
        .sort_values("rating", ascending=False)
        .iloc[0]
    )
    return movie_id_to_title[top["movieId"]]

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🎛 Controls")

selected_user = st.sidebar.selectbox(
    "Select User",
    ratings["userId"].unique()
)

num_recs = st.sidebar.slider("Recommendations", 6, 18, 12)

all_genres = sorted(set(
    g for gs in movies["genre_list"] for g in gs if g != "(no genres listed)"
))
selected_genres = st.sidebar.multiselect("Genres", all_genres)

selected_years = st.sidebar.slider(
    "Release Year",
    min_value=1900,
    max_value=2026,
    value=(1990, 2026)
)

# =========================
# HEADER
# =========================
st.markdown("""
<h1 style='text-align:center;'>🎬 CineMind Pro</h1>
<p style='text-align:center; font-size:18px;'>
AI-Powered Personalized Movie Recommendation Platform
</p>
<hr>
""", unsafe_allow_html=True)

# =========================
# RECOMMENDATIONS
# =========================
if st.button("🎯 Recommend Movies"):
    with st.spinner("Analyzing your taste profile... 🍿"):
        liked_movie = get_because_you_liked(selected_user)
        recs = recommend_movies(selected_user, num_recs)

        filtered = []
        for movie, score in recs:
            year = extract_year(movie)
            genres = movies[movies["title"] == movie]["genre_list"].values[0]

            if selected_genres and not any(g in genres for g in selected_genres):
                continue
            if not (selected_years[0] <= year <= selected_years[1]):
                continue

            filtered.append((movie, score))
            if len(filtered) >= num_recs:
                break

    max_score = max([s for _, s in filtered]) if filtered else 1

    st.subheader(f"✨ Recommended for you — because you liked **{liked_movie}**")

    rows = [filtered[i:i+3] for i in range(0, len(filtered), 3)]

    for row in rows:
        cols = st.columns(3)
        for col, (movie, score) in zip(cols, row):
            with col:
                info = get_movie_details(movie)
                genres = ", ".join(
                    movies[movies["title"] == movie]["genre_list"].values[0]
                )

                # ⭐ NORMALIZED RATING BETWEEN 3.5 - 5
                normalized_rating = round(3.5 + (score / max_score) * 1.5, 1)

                st.image(info["poster"], width=280)
                st.markdown(f"""
                <div style="text-align:center; font-weight:700;">
                    {movie}
                </div>
                <div style="text-align:center; font-size:12px; opacity:0.8;">
                    {info['release']} • {genres}
                </div>
                <div style="text-align:center; margin-top:6px;">
                    ⭐ {normalized_rating} / 5
                </div>
                """, unsafe_allow_html=True)

                with st.expander("More details"):
                    st.write(info["overview"])

# =========================
# FOOTER
# =========================
st.markdown("""
<hr>
<p style='text-align:center;'>
Built with ❤️ using Deep Learning, TMDB API & Streamlit<br>
</p>
""", unsafe_allow_html=True)

