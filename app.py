"""
Movie Recommender System - Flask Backend
Serves recommendations from the pre-trained cosine similarity model.
"""

from flask import Flask, request, jsonify, send_from_directory, after_this_request
import pickle
import pandas as pd
import os
import requests as req_lib

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

# Build a lower-case -> original title lookup for fuzzy matching
title_map = {t.lower(): t for t in movies["title"].tolist()}


def find_movie(query):
    q = query.strip().lower()
    if q in title_map:
        return title_map[q]
    for key, val in title_map.items():
        if q in key:
            return val
    return None


def recommend(movie_title, n=5):
    idx = movies[movies["title"] == movie_title].index[0]
    distances = sorted(
        enumerate(similarity[idx]), key=lambda x: x[1], reverse=True
    )
    results = []
    for i in distances[1:n + 1]:
        row = movies.iloc[i[0]]
        results.append({"title": row.title, "movie_id": int(row.movie_id)})
    return results


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
    """Proxy OMDb poster fetch server-side to avoid CORS issues."""
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"poster": None}), 400

    # Check cache first
    if title in _poster_cache:
        return jsonify({"poster": _poster_cache[title]})

    try:
        resp = req_lib.get(
            OMDB_URL,
            params={"apikey": OMDB_KEY, "t": title, "type": "movie"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=6,
        )
        data = resp.json()
        poster = data.get("Poster") or None
        if poster == "N/A":
            poster = None
        _poster_cache[title] = poster
        return jsonify({"poster": poster})
    except Exception as e:
        return jsonify({"poster": None, "error": str(e)})


@app.route("/static/<path:filename>")
def serve_static(filename):
    static_dir = os.path.join(BASE, "static")
    return send_from_directory(static_dir, filename)


@app.route("/api/movies", methods=["GET"])
def api_movies():
    return jsonify(movies["title"].tolist())


if __name__ == "__main__":
    app.run(debug=False, port=5000)
