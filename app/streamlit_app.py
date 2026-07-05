import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="SonicMatch", page_icon="🎵", layout="centered")

st.title("🎵 SonicMatch")
st.subheader("Find songs similar to your favorite track")
st.markdown("---")

song_name = st.text_input("Enter a song name", placeholder="e.g. Starboy")
artist_name = st.text_input("Artist name (optional)", placeholder="e.g. The Weeknd")
top_k = st.slider("Number of results", 3, 10, 5)

if st.button("Find Similar Songs", use_container_width=True):
    if song_name:
        with st.spinner("Finding similar songs..."):
            try:
                params = {
                    "song_name": song_name,
                    "top_k": top_k
                }
                if artist_name:
                    params["artist_name"] = artist_name

                response = requests.get(f"{API_URL}/search", params=params)

                if response.status_code == 200:
                    data = response.json()
                    results = data["results"]

                    if results:
                        st.success(f"Found {len(results)} similar songs for **{song_name}**")
                        st.markdown("---")

                        for i, song in enumerate(results):
                            with st.container():
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    st.markdown(f"### {i+1}. {song['track_name']}")
                                    st.markdown(f"**Artist:** {song['artists']}")
                                    st.markdown(f"**Genre:** {song['genres_text']}")
                                    st.markdown(f"💡 *{song['explanation']}*")

                                with col2:
                                    score = round(song['similarity_score'] * 100, 1)
                                    st.metric("Match", f"{score}%")

                                st.markdown("---")
                    else:
                        st.warning("No similar songs found.")

                elif response.status_code == 404:
                    st.error(f"Song '{song_name}' not found in the dataset. Try a different name.")
                else:
                    st.error(f"API error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Make sure the FastAPI server is running.")
    else:
        st.warning("Please enter a song name.")