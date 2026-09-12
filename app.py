"""
Movie Recommender System - Flask Backend
Serves recommendations from the pre-trained cosine similarity model.
"""

from flask import Flask, request, jsonify, send_from_directory, after_this_request
import pickle
import pandas as pd
import os
import requests as req_lib
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__, static_folder=".")

OMDB_KEY = "trilogy"  # Free OMDb API key
OMDB_URL = "http://www.omdbapi.com/"

# In-memory poster cache to avoid duplicate API calls
_poster_cache = {}

# Load model once at startup
BASE = os.path.dirname(os.path.abspath(__file__))

print("Loading movies.pkl ...", flush=True)
with open(os.path.join(BASE, "model", "movies.pkl"), "rb") as f:
    movies = pickle.load(f)

print("Loading similarity.pkl ... (this may take a few seconds)", flush=True)
with open(os.path.join(BASE, "model", "similarity.pkl"), "rb") as f:
    similarity = pickle.load(f)

print(f"Model ready. {len(movies)} movies loaded.", flush=True)

# Load popularity-sorted titles list for autocomplete
titles_path = os.path.join(BASE, "model", "titles.json")
if os.path.exists(titles_path):
    import json
    with open(titles_path, "r", encoding="utf-8") as f:
        autocomplete_titles = json.load(f)
else:
    autocomplete_titles = movies["title"].drop_duplicates().tolist()

# Build a unique lower-case -> original title lookup for fuzzy matching
title_map = {}
for t in autocomplete_titles:
    if t.lower() not in title_map:
        title_map[t.lower()] = t


def find_movie(query):
    q = query.strip().lower()
    if q in title_map:
        return title_map[q]
    for key, val in title_map.items():
        if q in key:
            return val
    return None


def recommend(movie_title, n=5):
    matching = movies[movies["title"] == movie_title]
    if matching.empty:
        return []
    idx = matching.index[0]
    
    # Fast cosine similarity using sparse linear kernel
    sim_scores = linear_kernel(similarity[idx], similarity).flatten()
    ranked_indices = sim_scores.argsort()[::-1]

    results = []
    seen = {movie_title.lower()}
    for i in ranked_indices:
        row = movies.iloc[i]
        t = row.title
        if t.lower() not in seen:
            seen.add(t.lower())
            raw_id = getattr(row, "movie_id", getattr(row, "imdb_id", i))
            try:
                m_id = int(raw_id)
            except Exception:
                m_id = str(raw_id)
            imdb_val = str(getattr(row, "imdb_id", ""))
            results.append({"title": t, "movie_id": m_id, "imdb_id": imdb_val})
            if len(results) == n:
                break
    return results


# Precompute title -> imdb_id lookup
title_to_imdb = {}
if "imdb_id" in movies.columns:
    for row in movies.itertuples():
        t = str(row.title).strip().lower()
        if t not in title_to_imdb and pd.notna(row.imdb_id):
            title_to_imdb[t] = str(row.imdb_id).strip()

# Session configured for IMDb requests
imdb_session = req_lib.Session()
imdb_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.imdb.com",
    "Referer": "https://www.imdb.com/",
})


def fetch_poster_by_imdb_id(imdb_id):
    """Fetch official poster image from IMDb (https://www.imdb.com/) using imdb_id."""
    if not imdb_id:
        return None

    # 1. Primary: IMDb CDN suggestion endpoint
    try:
        url = f"https://v3.sg.media-imdb.com/suggestion/x/{imdb_id}.json"
        resp = imdb_session.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("d", [])
            if items and "i" in items[0]:
                img_url = items[0]["i"].get("imageUrl")
                if img_url:
                    return img_url
    except Exception as e:
        print(f"IMDb direct lookup error for {imdb_id}: {e}", flush=True)

    # 2. Secondary: OMDb using IMDb ID (i=imdb_id)
    try:
        resp = req_lib.get(
            OMDB_URL,
            params={"apikey": OMDB_KEY, "i": imdb_id},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        data = resp.json()
        poster = data.get("Poster")
        if poster and poster != "N/A":
            return poster
    except Exception as e:
        print(f"OMDb lookup error for {imdb_id}: {e}", flush=True)

    return None


@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    query = request.args.get("movie", "").strip()
    if not query:
        return jsonify({"error": "No movie title provided."}), 400
    canonical = find_movie(query)
    if canonical is None:
        return jsonify({"error": f"Movie '{query}' not found in database."}), 404
    recs = recommend(canonical)
    return jsonify({"query": canonical, "recommendations": recs})


@app.route("/api/poster", methods=["GET"])
def api_poster():
    """Fetch poster from IMDb using imdb_id."""
    title = request.args.get("title", "").strip()
    imdb_id = request.args.get("imdb_id", "").strip()

    # Look up imdb_id from title if not explicitly provided
    if not imdb_id and title:
        canonical = find_movie(title) or title
        imdb_id = title_to_imdb.get(canonical.lower(), title_to_imdb.get(title.lower()))

    # Check cache first
    cache_key = imdb_id or title
    if cache_key and cache_key in _poster_cache:
        return jsonify({"poster": _poster_cache[cache_key], "imdb_id": imdb_id})

    poster = None
    if imdb_id:
        poster = fetch_poster_by_imdb_id(imdb_id)

    # Fallback to title query on OMDb if no poster found yet and title is present
    if not poster and title:
        try:
            resp = req_lib.get(
                OMDB_URL,
                params={"apikey": OMDB_KEY, "t": title, "type": "movie"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5,
            )
            data = resp.json()
            p = data.get("Poster")
            if p and p != "N/A":
                poster = p
        except Exception:
            pass

    if cache_key:
        _poster_cache[cache_key] = poster
    if title and title not in _poster_cache:
        _poster_cache[title] = poster

    return jsonify({"poster": poster, "imdb_id": imdb_id})


@app.route("/static/<path:filename>")
def serve_static(filename):
    static_dir = os.path.join(BASE, "static")
    return send_from_directory(static_dir, filename)


@app.route("/api/movies", methods=["GET"])
def api_movies():
    return jsonify(autocomplete_titles)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
