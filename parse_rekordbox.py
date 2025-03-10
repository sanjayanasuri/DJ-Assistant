from pathlib import Path
from bs4 import BeautifulSoup
import json

def save_tracks_to_json(tracks, output_path="rekordbox_tracks.json"):
    """Save extracted tracks as a JSON file for easy access."""
    data = [track.__dict__ for track in tracks]  # Convert objects to dicts
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    rekordbox_tracks = parse_rekordbox_xml("/Users/sanjayanasuri/Documents/rekordbox_tracks.xml")
    save_tracks_to_json(rekordbox_tracks)  # Save to JSON
    print(f"✅ Saved {len(rekordbox_tracks)} tracks to rekordbox_tracks.json")


class RekordboxTrack:
    """Class representing a track inside Rekordbox XML."""
    def __init__(self, track_id, title, artist, bpm, key, location):
        self.track_id = track_id
        self.title = title
        self.artist = artist
        self.bpm = bpm
        self.key = key
        self.location = location  # Filepath of the track
    
    def __repr__(self):
        return f"{self.title} - {self.artist} [{self.key}, {self.bpm} BPM]"

def parse_rekordbox_xml(xml_path):
    """Parse Rekordbox XML and extract track details."""
    with open(xml_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file.read(), "xml")

    tracks = []
    
    for track in soup.find_all("TRACK"):
        track_id = track.get("TrackID")
        title = track.get("Name", "Unknown Title")
        artist = track.get("Artist", "Unknown Artist")
        bpm = track.get("AverageBpm", "0.0")
        key = track.get("Tonality", "Unknown Key")
        location = track.get("Location", "")

        # Create a track object
        rekordbox_track = RekordboxTrack(track_id, title, artist, bpm, key, location)
        tracks.append(rekordbox_track)

    return tracks

if __name__ == "__main__":
    # Test parsing with your actual XML file
    rekordbox_tracks = parse_rekordbox_xml("/Users/sanjayanasuri/Documents/rekordbox_tracks.xml")
    for track in rekordbox_tracks[:10]:  # Show first 10 tracks
        print(track)

