import librosa
import numpy as np
import json

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)  
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    energy = np.mean(librosa.feature.rms(y=y))  
    
    return {
        "BPM": round(tempo),
        "Spectral_Centroid": spectral_centroid,
        "Energy": energy,
        "Chroma_Features": chroma.tolist()
    }

# Example usage
features = extract_audio_features("data/sample_track.mp3")
print(json.dumps(features, indent=4))

