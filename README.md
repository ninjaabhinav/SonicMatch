# SonicMatch 🎵
### Song Similarity Engine using Audio Features and Genre Embeddings

SonicMatch is a content-based music recommendation system that finds sonically similar songs given a track name. Unlike collaborative filtering approaches ("users who liked X also liked Y"), SonicMatch operates purely on the audio and semantic properties of the songs themselves — no user data, no play history, no black-box recommendations.

---

## How It Works

Each song is represented as a single vector combining two sources of signal:

1. **Audio features** (from Spotify's audio analysis API): danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo — scaled with `StandardScaler` and upweighted 3x to prevent genre embeddings from dominating by dimensionality alone.

2. **Genre embeddings**: the song's genre tags (e.g. "acoustic, indie-pop, singer-songwriter") are encoded into a 384-dimensional dense vector using `all-MiniLM-L6-v2` — a pretrained sentence-transformer encoder model running fully locally, with no LLM or generative API call anywhere in the pipeline.

These two vectors are concatenated into one `(393,)` vector per song. Similarity search is performed using **FAISS** (`IndexFlatIP` with cosine similarity on normalized vectors), returning the top-k nearest neighbors in milliseconds across 89,740 tracks.

---

## Dataset

- Source: Spotify Tracks Dataset (Kaggle) — 114,000 track-genre entries across 125 genres
- After deduplication: **89,740 unique tracks** with multi-genre label aggregation
- Tracks appearing under multiple genre labels had their genres combined into a single representation (e.g. a track tagged as both "acoustic" and "pop" is embedded as "acoustic, pop"), since most songs don't belong to a single genre

---

## Design Decisions

**Why sentence-transformers and not an LLM?**
Similarity search is a retrieval problem — embedding + nearest-neighbor search is the textbook-correct tool. LLMs are generative models; using one for retrieval adds latency, cost, and non-determinism without improving result quality. `all-MiniLM-L6-v2` is a BERT-family encoder that produces fixed-size deterministic vectors, runs locally, and requires no API key.

**Why upweight audio features 3x?**
The genre embedding is 384-dimensional vs 9-dimensional audio features. Without upweighting, cosine similarity is driven almost entirely by genre overlap, reducing the engine to a genre-matcher. Empirically tested at weights 1x, 3x, 5x, 10x — weight=3 produced the most musically coherent results, balancing genre context with sonic characteristics.

**Why FAISS IndexFlatIP over approximate indexes (IVF, HNSW)?**
At 89,740 vectors of dimension 393, exact search with `IndexFlatIP` runs in well under 100ms per query on CPU — fast enough for this use case without the accuracy tradeoff of approximate methods. HNSW/IVF would be the right call at 10M+ vectors.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Feature scaling | `scikit-learn` StandardScaler |
| Vector search | `FAISS` (IndexFlatIP, cosine similarity) |
| Backend API | `FastAPI` + `Uvicorn` |
| Frontend | `Streamlit` |
| Containerization | `Docker` + `Docker Compose` |

---

## Project Structure

```
sonic-match/
├── app/
│   ├── main.py              # FastAPI backend
│   └── streamlit_app.py     # Streamlit frontend
├── data/
│   ├── spotify_with_genres.csv
│   ├── song_vectors.npy
│   ├── songs_faiss.index
│   └── scaler.pkl
├── docker-compose.yml
└── requirements.txt
```

---

## Running Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start the FastAPI backend**
```bash
python -m uvicorn app.main:app --reload
```

**3. Start the Streamlit frontend** (in a separate terminal)
```bash
python -m streamlit run app/streamlit_app.py
```

**4. Open in browser**
- Streamlit UI: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search?song_name=X&artist_name=Y&top_k=5` | Find similar songs |
| GET | `/song/{song_name}` | Get audio features for a song |
| GET | `/` | Health check |

**Example response from `/search`:**
```json
{
  "query": "Starboy",
  "artist": "The Weeknd",
  "results": [
    {
      "track_name": "Verified Choppa 2",
      "artists": "Marksman",
      "genres_text": "dancehall, j-dance",
      "similarity_score": 0.9769,
      "explanation": "High match on instrumentalness (100.0%), energy (99.4%), acousticness (99.2%)"
    }
  ]
}
```

---

## What I'd Improve Next

- **Swap FAISS for AWS OpenSearch k-NN** for managed vector search at scale
- **Add an autoencoder** to learn a compressed joint representation from audio + genre features instead of manual concatenation
- **Expand to lyrics-based embeddings** via Genius API for richer semantic signal
- **Deploy on AWS ECS** with S3 for model artifact storage and CloudWatch for monitoring

---

## Author

Abhinav Mishra — [GitHub](https://github.com/ninjaabhinav)
