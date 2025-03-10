import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

load_dotenv()
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID,
                                                           client_secret=SPOTIFY_CLIENT_SECRET))

def get_track_features(track_id):
    features = sp.audio_features(track_id)[0]
    return {
        "BPM": features["tempo"],
        "Key": features["key"],
        "Danceability": features["danceability"],
        "Energy": features["energy"],
        "Valence": features["valence"]
    }

track_id = "TRACK_ID_HERE"
print(get_track_features(track_id))
