import streamlit as st
import pickle
import cv2
from deepface import DeepFace
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import random

# Set up Spotify API
CLIENT_ID = "aeae6a285a78433a8ef5d05cd573ed80"
CLIENT_SECRET = "e895916b8bed4fffb7718d8debe8b955"
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load preprocessed song data and inverted index
songs_df = pd.read_pickle("models/songs_with_top3_emotions.pkl")
with open("models/inverted_index.pkl", "rb") as f:
    inverted_index = pickle.load(f)

# Ensure inverted index keys are lowercase for proper matching
inverted_index = {k.lower(): v for k, v in inverted_index.items()}

# Normalize emotion names
emotion_mapping = {
    "joy": "happy", "amusement": "happy", "love": "happy", "gratitude": "happy",
    "sadness": "sad", "disappointment": "sad", "remorse": "sad", "grief": "sad",
    "anger": "angry",
    "fear": "fear",
    "surprise": "surprise",
    "neutral": "neutral"
}

# Helper function to fetch album cover
def get_song_album_cover(song_id):
    try:
        song_name, artist_name = song_id.split(" - ")
        results = sp.search(q=f"track:{song_name} artist:{artist_name}", type="track", limit=1)
        if results["tracks"]["items"]:
            track_info = results["tracks"]["items"][0]
            #Keep the [0]. It's necessary to select one image from the list before you access its url. If you remove it, your code will break.
            album_cover = track_info["album"]["images"][0]["url"]
            spotify_url = track_info["external_urls"]["spotify"]
            return album_cover, spotify_url
    except:
        pass
    return "https://i.postimg.cc/0QNxYz4V/social.png", None

# Streamlit UI
st.title(" Emotion-Based Music Recommender")
st.write(" Detecting your facial emotion...")

# Emotion detection
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    st.error(" Could not access webcam.")
else:
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    detected_emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
    normalized_emotion = emotion_mapping.get(detected_emotion.lower(), detected_emotion.lower())

    st.subheader(f" Detected Emotion: **{normalized_emotion.upper()}**")

    # Debugging print statement
    print(f"Detected Emotion: {normalized_emotion}")
    print(f"Available Emotions in Dataset: {list(inverted_index.keys())}")

    # Show dropdown with top emotions for user to select from
    available_emotions = list(inverted_index.keys())  
    emotion_options = [e for e in available_emotions if e != normalized_emotion]
    emotion_choice = normalized_emotion
    
    # Handle missing "happy" by mapping it to "joy" or "amusement"
    if emotion_choice.lower() == "happy":
        if "joy" in inverted_index:
            emotion_choice = "joy"
        elif "amusement" in inverted_index:
            emotion_choice = "amusement"
    elif emotion_choice.lower() == "sad":
            # Choose the most populated sad-related emotion
        if "sadness" in inverted_index:
            emotion_choice = "sadness"
        elif "remorse" in inverted_index:
            emotion_choice = "remorse"
        elif "disappointment" in inverted_index:
            emotion_choice = "disappointment"
    elif emotion_choice.lower() == "angry":
        if "anger" in inverted_index:
            emotion_choice = "anger"
        elif "annoyance" in inverted_index:
            emotion_choice = "annoyance"
        elif "disapproval" in inverted_index:
            emotion_choice = "disapproval"

    # Slider to choose number of recommendations
    num_songs = st.slider(" Select number of recommendations:", min_value=5, max_value=30, value=5)

    # Recommend songs
    st.markdown(" **Recommended Songs:**")

    # Debugging print statement
    print(f"Selected Emotion for Recommendations: {emotion_choice.lower()}")

    if emotion_choice.lower() not in inverted_index:
        st.warning(f"No songs found for `{emotion_choice}`. Try another emotion!")
    else:
        recommendations = inverted_index[emotion_choice.lower()]
        random.shuffle(recommendations)  # Shuffle songs every time
        sorted_recommendations = recommendations[:num_songs]

        for song_id, score in sorted_recommendations:
            cover_url, spotify_url = get_song_album_cover(song_id)
            st.image(cover_url, width=150)
            if spotify_url:
                st.markdown(f"**🎵 [{song_id}]({spotify_url})** — _Confidence Score: {score}_", unsafe_allow_html=True)
            else:
                st.write(f"**🎵 {song_id}** — _Confidence Score: {score}_")

st.caption("Refresh the page to detect a new emotion!")
