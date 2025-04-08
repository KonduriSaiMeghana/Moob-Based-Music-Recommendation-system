# process_emotions.py
import pandas as pd
from transformers import pipeline
import pickle

# Load dataset
df = pd.read_csv("C://Users//kondu//OneDrive//Desktop//6thsem//appli//app//app//data//spotify_millsongdata.csv")
df.dropna(subset=['text'], inplace=True)
df['text'] = df['text'].fillna("")

# Load emotion classifier
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", return_all_scores=True)

# Extract top 3 emotions
def get_top_3_emotions(text):
    text = text[:500]
    predictions = emotion_classifier(text)[0]
    sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
    top3 = sorted_preds[:3]
    return [(p['label'], round(p['score'], 3)) for p in top3]

# Apply to dataset
df['top3_emotions'] = df['text'].apply(get_top_3_emotions)

# Create inverted index
inverted_index = {}

for idx, row in df.iterrows():
    song_id = f"{row['song']} - {row['artist']}"
    for emotion, score in row['top3_emotions']:
        if emotion not in inverted_index:
            inverted_index[emotion] = []
        inverted_index[emotion].append((song_id, score))

# Save
df.to_pickle("models/songs_with_top3_emotions.pkl")
with open("models/inverted_index.pkl", "wb") as f:
    pickle.dump(inverted_index, f)

print("✅ Saved top 3 emotions and inverted index to /models folder.")
