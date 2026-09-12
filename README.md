# 🎬 CineMatch — AI Movie Recommender System

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.x-green?style=for-the-badge)
![TMDB](https://img.shields.io/badge/TMDB-Dataset-01d277?style=for-the-badge&logo=themoviedatabase&logoColor=white)
![Git LFS](https://img.shields.io/badge/Git_LFS-Enabled-orange?style=for-the-badge&logo=git-lfs&logoColor=white)

> A modern, content-based movie recommender system powered by natural language processing and cosine similarity. Features an interactive, cinematic web interface with real-time autocomplete, dynamic poster fetching, and smooth video background.

---

## ✨ Features

- **🧠 Content-Based Filtering Model**: Recommends the top 5 most similar films based on genres, keywords, cast, crew (director), and overview text.
- **⚡ Fast Cosine Similarity Matrix**: Vectorized textual tags with Bag-of-Words / CountVectorizer and Porter Stemming for high-precision semantic matching.
- **🎨 Cinematic Web Interface**:
  - Ambient looping background video with film grain and sprocket strips.
  - Elegant typography using *Cinzel* and *Cormorant Garamond*.
  - Rich interactive recommendation cards with Roman numeral ranks (Pick I to V), shimmer loading skeletons, and hover zoom & glow effects.
  - Instant autocomplete search with fuzzy matching across 4,800+ films.
  - One-click popular movie chips for quick discovery.
- **🖼️ Real-Time Poster Fetching**: Integrates poster image retrieval with in-memory server-side caching.
- **📱 Responsive Design**: Seamlessly adapts across desktops, tablets, and mobile devices.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3 (Vanilla CSS, Glassmorphism, CSS Grid), Vanilla JavaScript |
| **Backend** | Python 3, Flask, Requests |
| **ML & NLP** | Scikit-Learn (`CountVectorizer`, `cosine_similarity`), NLTK (`PorterStemmer`), Pandas, NumPy |
| **Dataset** | TMDB 5000 Movie & Credits Dataset |
| **Storage / Models** | Pickle (`movies.pkl`, `similarity.pkl`), Git LFS |

---

## 📂 Project Structure

```text
Movie-Recommender-System/
├── Dataset/
│   ├── tmdb_5000_credits.csv   # Cast, crew, and production credits
│   └── tmdb_5000_movies.csv    # Movie metadata (genres, budget, keywords, etc.)
├── model/
│   ├── movies.pkl              # Cleaned movie metadata DataFrame
│   ├── similarity.pkl          # Cosine similarity matrix (Git LFS)
│   └── titles.json             # Cached titles list for autocomplete
├── notebook/
│   └── movie.ipynb             # Jupyter Notebook with full EDA, NLP & model training
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

1. **Feature Extraction**: Extracts `genres`, `keywords`, top 3 `cast` members, `director` from crew, and the plot `overview`.
2. **Text Normalization**:
   - Converts all words to lowercase and removes spaces between multi-word tags (e.g., `Sam Worthington` → `samworthington`) to prevent entity collision.
   - Combines all metadata into a single unified `tags` feature column.
3. **Stemming & Vectorization**:
   - Applies NLTK's `PorterStemmer` (e.g., `actions`, `action` → `action`).
   - Converts text into 5,000 top-frequency vectors using `CountVectorizer(max_features=5000, stop_words='english')`.
4. **Cosine Similarity**:
   - Measures the angular distance between movie vectors in high-dimensional space:
   $$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
   - Returns the top 5 closest neighbors for any input title.

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
pip install flask pandas scikit-learn requests
```

### 4. Background Video (Optional)

To enable the ambient video background:
1. Place any `.mp4` video file inside the `static/` folder.
2. Name it `bg.mp4` (or update the `<source src="...">` path in `index.html`).

### 5. Run the Application

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

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
    { "movie_id": 19995, "title": "Aliens vs Predator: Requiem" },
    { "movie_id": 1858, "title": "Aliens" },
    { "movie_id": 679, "title": "Falcon Rising" },
    { "movie_id": 106, "title": "Independence Day" },
    { "movie_id": 348, "title": "Titan A.E." }
  ]
}
```

### 2. Fetch Movie Poster
- **Endpoint**: `/api/poster`
- **Method**: `GET`
- **Query Parameter**: `title` *(string, required)*
- **Response**:
```json
{
  "poster": "https://m.media-amazon.com/images/M/...jpg"
}
```

### 3. Movie Title Autocomplete List
- **Endpoint**: `/api/movies`
- **Method**: `GET`
- **Response**: Array of all 4,806 movie titles.

---

## 👨‍💻 Author

- **Ayush** — [@Ayu-this-side](https://github.com/Ayu-this-side)
