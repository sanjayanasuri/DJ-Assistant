import time
import logging
import pandas as pd
from fetch_spotify import SpotifyAuthManager, search_spotify_track
from parse_rekordbox import fetch_rekordbox_data
from recommend_tracks import TrackRecommender
from extract_features import get_spotify_audio_features_batch
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Credentials
SPOTIFY_CLIENT_ID = "your_spotify_client_id"
SPOTIFY_CLIENT_SECRET = "your_spotify_client_secret"
SPOTIFY_REDIRECT_URI = "http://localhost:8888/callback"
OPENAI_API_KEY = "your_openai_api_key"

# Initialize API Clients
spotify_auth = SpotifyAuthManager(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Load Rekordbox Data
rekordbox_file = "data/rekordbox_tracks.xml"
df = fetch_rekordbox_data(rekordbox_file)
if df.empty:
    logging.error("No valid tracks found in Rekordbox XML.")
    exit()

logging.info(f"Loaded {len(df)} tracks from Rekordbox.")

# Normalize Titles with OpenAI
def normalize_titles(titles, client):
    normalized_titles = []
    for title in titles:
        prompt = f"Normalize this song title and extract the artist. Format: 'Title - Artist': {title}"
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            normalized_titles.append(response.choices[0].message.content.strip())
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            normalized_titles.append(title)
        time.sleep(1)
    return normalized_titles

df['cleaned_title'] = normalize_titles(df['title'].tolist(), openai_client)
logging.info("Track titles normalized using OpenAI.")

# Match with Spotify
matches = []
for _, row in df.iterrows():
    title, artist = row['cleaned_title'].split(' - ', 1) if ' - ' in row['cleaned_title'] else (row['cleaned_title'], "Unknown Artist")
    spotify_id, spotify_url = search_spotify_track(title, artist, spotify_auth.sp)
    matches.append({
        'rekordbox_id': row['id'],
        'spotify_id': spotify_id,
        'spotify_url': spotify_url
    })
    if spotify_id:
        logging.info(f"Matched {title} by {artist} to Spotify track ID {spotify_id}")
    else:
        logging.warning(f"No match found for {title} by {artist}")

match_df = pd.DataFrame(matches)
df = df.merge(match_df, left_on='id', right_on='rekordbox_id', how='left')

# Get Audio Features
audio_features = get_spotify_audio_features_batch(df['spotify_id'].dropna().tolist(), spotify_auth)
if audio_features:
    features_df = pd.DataFrame(audio_features)
    df = df.merge(features_df, left_on='spotify_id', right_on='id', how='left')
    logging.info("Audio features successfully merged.")
else:
    logging.warning("No audio features retrieved from Spotify.")

# Generate Track Recommendations
recommender = TrackRecommender()
df['recommended_tracks'] = df.apply(lambda row: recommender.get_recommendations(row), axis=1)

# Save Processed Data
df.to_csv("data/processed_tracks.csv", index=False)
logging.info("Processed track data saved to data/processed_tracks.csv.")

print("Final Data:")
print(df[['title', 'artist', 'recommended_tracks']].head())
