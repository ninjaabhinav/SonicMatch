from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import faiss
import joblib

app = FastAPI(title="SonicMatch API", description="Song similarity engine using audio features and genre embeddings")

# --- load everything once at startup ---
df = pd.read_csv("data/spotify_with_genres.csv")
song_vectors = np.load("data/song_vectors.npy").astype("float32")
index = faiss.read_index("data/songs_faiss.index")
scaler = joblib.load("data/scaler.pkl")

# normalize vectors
norms = np.linalg.norm(song_vectors, axis=1, keepdims=True)
norms = np.where(norms == 0, 1.0, norms)
normalized_vectors = song_vectors / norms

continuous_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']


# --- helper functions ---

def explain_match(query_idx, result_idx):
    query_features = df.loc[query_idx, continuous_cols]
    result_features = df.loc[result_idx, continuous_cols]

    abs_diff = abs(query_features - result_features)

    similarity_pct = {}
    for col in continuous_cols:
        score = max(0, 1 - abs_diff[col] / 3)
        similarity_pct[col] = round(score * 100, 1)

    top_features = sorted(similarity_pct.items(), key=lambda x: x[1], reverse=True)[:3]
    explanation = "High match on " + ", ".join([f"{feat} ({pct}%)" for feat, pct in top_features])

    return explanation


def find_similar(song_name, artist_name=None, top_k=5):
    matches = df[df['track_name'].str.lower() == song_name.lower()]

    if artist_name and not matches.empty:
        matches = matches[matches['artists'].str.lower().str.contains(artist_name.lower(), na=False)]

    if matches.empty:
        return None

    idx = matches.index[0]
    query_vector = normalized_vectors[idx].reshape(1, -1)

    distances, indices = index.search(query_vector, top_k + 10)

    result_indices = indices[0][1:]
    result_scores = distances[0][1:]

    results_df = df.iloc[result_indices].copy()
    results_df['similarity_score'] = result_scores
    results_df = results_df.drop_duplicates(subset=['track_name', 'artists'])

    # remove query song itself from results
    query_track = df.loc[idx, 'track_name'].lower()
    query_artist = df.loc[idx, 'artists'].lower()
    results_df = results_df[
        ~((results_df['track_name'].str.lower() == query_track) &
          (results_df['artists'].str.lower() == query_artist))
    ]

    results_df = results_df.head(top_k)

    results_df['explanation'] = [
        explain_match(idx, result_idx)
        for result_idx in results_df.index
    ]

    return results_df[['track_name', 'artists', 'genres_text', 'similarity_score', 'explanation']].reset_index(drop=True)


# --- endpoints ---

@app.get("/")
def root():
    return {"message": "SonicMatch API is running"}


@app.get("/search")
def search(song_name: str, artist_name: str = None, top_k: int = 5):
    results = find_similar(song_name, artist_name, top_k)

    if results is None:
        raise HTTPException(status_code=404, detail=f"Song '{song_name}' not found in dataset")

    return {
        "query": song_name,
        "artist": artist_name,
        "results": results.to_dict(orient="records")
    }


@app.get("/song/{song_name}")
def song_info(song_name: str):
    matches = df[df['track_name'].str.lower() == song_name.lower()]

    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Song '{song_name}' not found")

    song = matches.iloc[0]

    return {
        "track_name": song['track_name'],
        "artists": song['artists'],
        "album_name": song['album_name'],
        "genres": song['genres_text'],
        "popularity": int(song['popularity']),
        "audio_features": {
            col: round(float(song[col]), 4) for col in continuous_cols
        }
    }