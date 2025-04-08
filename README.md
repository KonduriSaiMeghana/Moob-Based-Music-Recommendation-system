#  Mood-Based Music Recommendation System

This project is a Music recommendation system that detects a user's facial emotion in real-time and recommends songs by analyzing song lyrics using **Natural Language Processing (NLP)**.

It combines **Computer Vision**, **Transformers (BERT)**, and the **Spotify API** to create a personalized and emotionally intelligent music experience.

---

##  Features

-  Real-time facial emotion detection using webcam and DeepFace  
-  NLP-based emotion classification on song lyrics using BERT  
-  Song recommendations based on emotional similarity  
-  Album cover & Spotify link integration via Spotify API  
-  Web interface built with Streamlit  

---

## Dataset Used

🎧 **Spotify Million Song Dataset**  
Includes: song name, artist, lyrics, and Spotify link

---

## Technologies Used

| Component               | Technology                        |
|------------------------|-----------------------------------|
| Emotion Detection       | DeepFace + OpenCV (Webcam Input) |
| Lyrics Emotion Analysis | BERT (GoEmotions via Transformers) |
| Web Interface           | Streamlit                         |
| Music Data              | Spotify Million Song Dataset      |
| API Integration         | Spotipy (Spotify API)             |
| Data Storage            | Pandas, Pickle                    |

---

## Process Workflow

This project involves two main stages:
1. **Preprocessing Song Data**
2. **Real-Time Emotion-Based Recommendation**

### 🔹 A. Song Lyrics Preprocessing (`process_emotions.py`)

> This step is done **before** running the app. It prepares all song data for recommendations.

#### Steps:
- Load a dataset of 10,000 songs from the Spotify Million Song Dataset  
- Clean the lyrics and remove empty or missing text  
- Use a BERT-based NLP model (`bert-base-go-emotion`) to classify emotions in each song's lyrics  
- Extract the **top 3 emotions** per song with confidence scores  
- Build an **inverted index** that maps each emotion to songs  
- Save the processed song data and emotion index to `.pkl` files


### 🔹 B. Real-Time Emotion Detection & Recommendation (`app.py`)

> This is the Streamlit app the user interacts with.

#### Steps:
- Capture webcam input using OpenCV  
- Analyze the image using DeepFace to detect the user's facial emotion  
- Map detected emotion to simplified emotion categories (e.g., “joy” → “happy”)  
- Fetch songs from the inverted index that match the detected emotion  
- Use Spotify API to get:
  - Album cover images  
  - Links to play the song on Spotify  
- Display song recommendations with:
  - Album cover  
  - Song name & artist  
  - Confidence score

---


