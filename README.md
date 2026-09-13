# 🎬 CineMatch — AI Movie Recommender System

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.x-green?style=for-the-badge)
![IMDb](https://img.shields.io/badge/IMDb-Posters-F5C518?style=for-the-badge&logo=imdb&logoColor=black)
![Vercel](https://img.shields.io/badge/Vercel-Deployed-black?style=for-the-badge&logo=vercel&logoColor=white)
![Live Demo](https://img.shields.io/badge/Live_Demo-Visit_Website-00C7B7?style=for-the-badge&logo=vercel&logoColor=white)

> A modern, content-based movie recommender system powered by natural language processing and cosine similarity. Features an interactive, cinematic web interface with real-time autocomplete, dynamic IMDb poster fetching, and smooth video background.

🌐 **Live Deployed Website**: [https://movie-recommender-system-y8a7.vercel.app/](https://movie-recommender-system-y8a7.vercel.app/)

---

## ✨ Features

- **🧠 Content-Based Filtering Model**: Recommends the top 5 most similar films based on genres, overview, and tagline across a comprehensive library of 45,000+ films.
- **⚡ Fast TF-IDF Similarity Engine**: Vectorized textual tags with unigrams & bigrams (`ngram_range=(1,2)`), 50,000 features, and NLTK WordNet lemmatization for high-precision semantic matching in ~20–35ms.
- **🎨 Cinematic Web Interface**:
  - Ambient looping background video with film grain and sprocket strips.
  - Elegant typography using *Cinzel* and *Cormorant Garamond*.
  - Rich interactive recommendation cards with Roman numeral ranks (Pick I to V), shimmer loading skeletons, and hover zoom & glow effects.
  - Instant autocomplete search with prefix-ranking across 42,000+ unique films.
  - One-click popular movie chips for quick discovery.
- **🖼️ Real-Time IMDb Poster Fetching**: Integrates high-resolution poster image retrieval directly from IMDb via `imdb_id` with in-memory server-side caching.
- **📱 Responsive Design**: Seamlessly adapts across desktops, tablets, and mobile devices.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3 (Vanilla CSS, Glassmorphism, CSS Grid), Vanilla JavaScript |
| **Backend** | Python 3, Flask, Requests |
| **ML & NLP** | Scikit-Learn (`TfidfVectorizer`, `linear_kernel`), NLTK (`WordNetLemmatizer`, `stopwords`), Pandas, NumPy |
| **Dataset** | The Movies Metadata Dataset (`movies_metadata.csv` — 45,000+ films) |
| **Storage / Models** | Pickle (`movies.pkl`, `similarity.pkl`), JSON (`titles.json`), Git LFS |

---

## 📂 Project Structure

```text
Movie-Recommender-System/
├── Dataset/
│   └── movies_metadata.csv     # Movie metadata (genres, titles, overviews, taglines)
├── model/
│   ├── movies.pkl              # Cleaned movie metadata DataFrame
│   ├── similarity.pkl          # TF-IDF sparse similarity model (Git LFS)
│   └── titles.json             # Cached titles list for autocomplete ranked by popularity
├── notebook/
│   └── movieRecommender.ipynb  # Jupyter Notebook with full EDA, NLP & model training
├── static/
│   ├── .gitkeep
│   └── bg.mp4                  # Ambient background video (place your video here)
├── .gitattributes              # Git LFS tracking configuration
├── .gitignore                  # Git ignore rules (virtualenv, cache, large media)
├── app.py                      # Flask application and REST API endpoints
├── index.html                  # Single-page cinematic frontend UI
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## 🧠 How the Recommendation Engine Works

1. **Feature Extraction**: Combines the plot `overview`, parsed `genres`, and `tagline` into a single, unified `tags` feature column.
2. **Text Normalization & Lemmatization**:
   - Converts all words to lowercase and removes punctuation using regex.
   - Filters out English stopwords.
   - Applies NLTK's `WordNetLemmatizer` (e.g., `running`, `runs` → `run`) for accurate morphological normalization.
3. **TF-IDF Vectorization**:
   - Extracts unigram and bigram tokens with `TfidfVectorizer(max_features=50000, ngram_range=(1,2), stop_words='english')`.
   - Generates a compact scipy sparse feature matrix (`45,426 × 50,000`).
4. **Cosine Similarity**:
   - Computes cosine similarity against the query movie vector using `linear_kernel`:
   $$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
   - Sorts candidates in descending similarity order and returns the top 5 closest kindred films in ~20ms.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git & Git LFS ([Download Git LFS](https://git-lfs.com/))

### 1. Clone the Repository

```bash
git clone https://github.com/Ayu-this-side/Movie-Recommender-System.git
cd Movie-Recommender-System
```

> **Note**: Make sure to pull Git LFS objects to download `model/similarity.pkl`:
> ```bash
> git lfs pull
> ```

### 2. Create and Activate Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install flask pandas scikit-learn requests nltk
```

### 4. Background Video

A web-optimized, seamless looping starry background video is included at `static/bg.mp4`. It is encoded for fast web streaming with `+faststart` and instant autoplay.

### 5. Run Locally

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 6. Live Deployment (Vercel)

The project is deployed on Vercel:
- **Production URL**: [https://movie-recommender-system-y8a7.vercel.app/](https://movie-recommender-system-y8a7.vercel.app/)

---

## 🔌 API Reference

### 1. Get Recommendations
- **Endpoint**: `/api/recommend`
- **Method**: `GET`
- **Query Parameter**: `movie` *(string, required)*
- **Response**:
```json
{
  "query": "Avatar",
  "recommendations": [
    { "movie_id": 26535, "imdb_id": "tt1630029", "title": "Avatar 2" },
    { "movie_id": 26541, "imdb_id": "tt3501632", "title": "Thor: Ragnarok" },
    { "movie_id": 13880, "imdb_id": "tt0972558", "title": "The Inhabited Island" },
    { "movie_id": 43412, "imdb_id": "tt3705822", "title": "Moontrap: Target Earth" },
    { "movie_id": 14116, "imdb_id": "tt0024663", "title": "The Three Musketeers" }
  ]
}
```

### 2. Fetch Movie Poster
- **Endpoint**: `/api/poster`
- **Method**: `GET`
- **Query Parameters**: `title` *(string)* or `imdb_id` *(string)*
- **Response**:
```json
{
  "imdb_id": "tt0468569",
  "poster": "https://m.media-amazon.com/images/M/MV5BMTMxNTMwODM0NF5BMl5BanBnXkFtZTcwODAyMTk2Mw@@._V1_.jpg"
}
```

### 3. Movie Title Autocomplete List
- **Endpoint**: `/api/movies`
- **Method**: `GET`
- **Response**: Array of 42,263 unique movie titles ranked by popularity.

---

## 👨‍💻 Author

- **Ayush** — [@Ayu-this-side](https://github.com/Ayu-this-side)
