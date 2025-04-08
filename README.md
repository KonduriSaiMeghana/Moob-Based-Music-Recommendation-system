# Mood-Based-Music-Recommendation-system

This project is an **Music recommendation system** that detects a user's **facial emotion** in real-time and recommends songs by analyzing **song lyrics** using **Natural Language Processing (NLP)**.

It combines **Computer Vision**, **Transformers (BERT)**, and the **Spotify API** to create a personalized and emotionally intelligent music experience.

#  Features

-  Real-time facial emotion detection using webcam and DeepFace
-  NLP-based emotion classification on song lyrics using BERT
-  Song recommendations based on emotional similarity
-  Album cover & Spotify link integration via Spotify API
-  Web interface built with Streamlit

# Dataset Used

🎧 Spotify Million Song Dataset 
Includes: song name, artist, lyrics, and Spotify link

#  Technologies Used

| Component               | Technology                        |
|-------------------------|-----------------------------------|
| Emotion Detection       | DeepFace + OpenCV (Webcam Input) |
| Lyrics Emotion Analysis | BERT (GoEmotions via Transformers) |
| Web Interface           | Streamlit                         |
| Music Data              | Spotify Million Song Dataset      |
| API Integration         | Spotipy (Spotify API)             |
| Data Storage            | Pandas, Pickle                    |

# Process Workflow
This project involves two main stages: Preprocessing Song Data and Real-Time Emotion-Based Recommendation.


**A. Song Lyrics Preprocessing (process_emotions.py):**
This step is done before running the main app. It prepares all song data for recommendations.

**Steps:**
1. Load a dataset of 10,000 songs from the Spotify Million Song Dataset.

2. Clean the lyrics and remove empty or missing text.

3. Use a BERT-based NLP model (bert-base-go-emotion) to classify emotions in each song's lyrics.
4. Extract the top 3 emotions per song with confidence scores.
5. Build an inverted index that maps:
6. Save the processed song data and emotion index to .pkl files (used by the main app).

**B. Real-Time Emotion Detection & Recommendation (app.py):**
This is the Streamlit app the user interacts with.

**Steps:**
1. Capture webcam input using OpenCV.

2. Analyze the image using DeepFace to detect the user's facial emotion (e.g., happy, sad, angry).

3. Map detected emotion to the simplified emotion categories used in the song index (e.g., “joy” → “happy”).

4. Fetch songs from the inverted index that match the detected emotion.

5. Use Spotify API to get:

- Album cover images
- Links to play the song on Spotify

6. Display song recommendations with:

- Album cover
- Song name & artist

