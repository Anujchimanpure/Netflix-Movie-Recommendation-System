import streamlit as st
import pandas as pd
import pickle
import requests

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Netflix Recommender",
    layout="wide"
)

# ================= CINEMATIC NETFLIX CSS =================
st.markdown("""
<style>

/* Background */
.stApp {
    background: radial-gradient(circle at top, #1c1c1c 0%, #0b0b0b 60%);
    color: white;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* HERO SECTION */
.hero {
    padding: 80px 40px;
    background: linear-gradient(to bottom, rgba(0,0,0,0.3), #0b0b0b),
                url("https://images.unsplash.com/photo-1608178398319-48f814d0750c");
    background-size: cover;
    border-radius: 20px;
    margin-bottom: 40px;
}

.hero h1 {
    font-size: 56px;
    font-weight: 900;
    color: #e50914;
    text-shadow: 0 0 30px rgba(229,9,20,0.7);
}

.hero p {
    font-size: 20px;
    color: #e5e5e5;
    max-width: 700px;
}

/* Input */
input {
    background-color: #1f1f1f !important;
    color: white !important;
    border-radius: 10px;
    border: 1px solid #333;
    padding: 12px;
}

/* Button */
button[kind="primary"] {
    background: linear-gradient(135deg, #e50914, #b20710) !important;
    color: white !important;
    font-size: 16px;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 24px;
    box-shadow: 0 0 20px rgba(229,9,20,0.6);
}

/* Section titles */
.section-title {
    font-size: 32px;
    font-weight: 800;
    margin: 40px 0 20px 0;
}

/* Movie Card */
.movie-card {
    background: #141414;
    border-radius: 16px;
    padding: 15px;
    box-shadow: 0 0 25px rgba(0,0,0,0.6);
}

.movie-title {
    font-size: 20px;
    font-weight: 700;
    margin-top: 10px;
}

.movie-desc {
    font-size: 14px;
    color: #b3b3b3;
}

</style>
""", unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<div class="hero">
    <h1>Netflix Movie Recommendation System</h1>
    <p>
        Discover movies you’ll love — powered by Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
df = pd.read_csv("netflix_clean_data.csv")
df["title_clean"] = df["title"].str.lower().str.strip()
df["description"] = df["description"].fillna("Description not available.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def compute_similarity(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['description'])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

similarity = compute_similarity(df)


# ================= OMDb =================
OMDB_API_KEY = "bbe50b43"

def fetch_poster(title):
    try:
        url = "http://www.omdbapi.com/"
        params = {
            "t": title,
            "apikey": OMDB_API_KEY
        }

        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if data.get("Response") == "True":
            poster = data.get("Poster")
            if poster and poster != "N/A":
                return poster
    except:
        return None

    return None
    
@st.cache_data
def get_poster_cached(title):
    return fetch_poster(title)


# ================= RECOMMENDER =================
def recommend_movie(title):
    title = title.lower().strip()

    matches = df[df["title_clean"].str.contains(title)]
    
    if matches.empty:
        return None, None
    
    idx = matches.index[0]

    scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)

    recs = []
    for i in scores[1:7]:
        recs.append({
            "title": df.iloc[i[0]]["title"],
            "description": df.iloc[i[0]]["description"]
        })

    return recs, df.iloc[idx]["description"]

# ================= INPUT (SIMPLE TEXT ONLY) =================
st.markdown("## 🔍 Search a Movie")

movie_name = st.text_input(
    "",
    placeholder="Type the exact movie name (e.g. Iron Man)"
)

submit = st.button("Recommend")

# ================= OUTPUT =================
if submit:
    recs, desc = recommend_movie(movie_name)

    if recs is None:
        st.error("Movie not found. Please check spelling.")
    else:
        st.markdown('<div class="section-title">🎬 Selected Movie</div>', unsafe_allow_html=True)
        st.write(desc)

        st.markdown('<div class="section-title">🔥 Recommended For You</div>', unsafe_allow_html=True)

        cols = st.columns(5)

        for idx, movie in enumerate(recs):
            with cols[idx % 5]:
                poster = get_poster_cached(movie["title"])
        
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        
                # ---- FIXED POSTER AREA ----
                if poster:
                    st.markdown(
                        f"""
                        <div style="height:360px; overflow:hidden; border-radius:12px;">
                            <img src="{poster}" style="width:100%; height:100%; object-fit:cover;">
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div style="
                            height:360px;
                            background-color:#222;
                            border-radius:12px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            color:#b3b3b3;
                            font-size:14px;
                        ">
                            Poster not available
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
                # ---- DETAILS BELOW POSTER (ALWAYS SAME POSITION) ----
                st.markdown(
                    f'<div class="movie-title">{movie["title"]}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="movie-desc">{movie["description"][:120]}...</div>',
                    unsafe_allow_html=True
                )
        
                st.markdown('</div>', unsafe_allow_html=True)


st.markdown("""
<hr style="margin-top:60px; border: 0.5px solid #222;">

<div style="
    text-align: center;
    color: #777;
    font-size: 13px;
    padding: 20px 10px 10px 10px;
    line-height: 1.6;
">
    ⚠️ This recommendation model is trained on a limited dataset. - Some movies or posters may not appear.<br>
    This project is built for learning and demonstration purposes.
</div>
""", unsafe_allow_html=True)

