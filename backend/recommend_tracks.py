import faiss
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import logging
from scripts.fetch_tracks import fetch_all_tracks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NCF(nn.Module):
    """Neural Collaborative Filtering model for DJ track recommendations"""
    def __init__(self, num_tracks: int, num_factors: int = 20, layers: List[int] = [40, 20]):
        super(NCF, self).__init__()

        # Validate input parameters
        if num_tracks <= 0:
            raise ValueError("num_tracks must be positive")

        self.num_tracks = num_tracks
        self.track_embedding = nn.Embedding(num_tracks, num_factors)
        self.context_embedding = nn.Embedding(num_tracks, num_factors)

        # MLP layers with smaller dimensions
        self.fc_layers = nn.ModuleList()
        input_size = num_factors * 2

        for size in layers:
            self.fc_layers.append(nn.Linear(input_size, size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(p=0.1))  # Reduced dropout
            input_size = size

        self.output_layer = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, track_indices: torch.Tensor, context_indices: torch.Tensor) -> torch.Tensor:
        # Input validation
        if torch.any(track_indices >= self.num_tracks) or torch.any(context_indices >= self.num_tracks):
            raise ValueError("Track indices out of bounds")

        track_embed = self.track_embedding(track_indices)
        context_embed = self.context_embedding(context_indices)

        x = torch.cat([track_embed, context_embed], dim=1)

        for layer in self.fc_layers:
            x = layer(x)

        x = self.output_layer(x)
        return self.sigmoid(x)

    def save_model(self, path: str):
        """Save model state with error handling"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.state_dict(), path)
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: str):
        """Load model state with error handling"""
        try:
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

def build_faiss_index(tracks: List[dict], feature_dims: int = 7) -> Tuple:
    """Build FAISS index for similarity search using multiple features"""
    logging.info("Building FAISS index...")
    index = faiss.IndexFlatL2(feature_dims)
    track_vectors = []
    track_ids = []

    for track in tracks:
        # Create feature vector using all our features
        vec = np.array([
            track["danceability"],
            track["energy"],
            track["bpm"] / 180.0,  # Normalize BPM
            track["key_compatibility"],
            track["intensity"],
            track["groove"],
            track["transition_score"]
        ], dtype=np.float32)

        index.add(np.array([vec]))
        track_vectors.append(vec)
        track_ids.append(track["id"])

    logging.info(f"FAISS index built with {len(track_ids)} tracks")
    return index, np.array(track_vectors), np.array(track_ids)

def generate_training_data(tracks: List[dict], num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data for NCF model with memory efficiency"""
    logging.info(f"Generating {num_samples} training samples...")
    num_tracks = len(tracks)
    track_ids = [track["id"] for track in tracks]

    X = []  # Input pairs
    y = []  # Labels
    samples_per_batch = min(100, num_samples // 10)  # Process in smaller batches

    for batch in range(0, num_samples, samples_per_batch):
        batch_X = []
        batch_y = []

        for _ in range(min(samples_per_batch, num_samples - batch)):
            # Positive samples (tracks that work well together)
            track1_idx = random.randint(0, num_tracks - 1)
            track1 = tracks[track1_idx]

            # Find compatible tracks (within BPM and key range)
            compatible_tracks = []
            for i, t in enumerate(tracks):
                if i != track1_idx:
                    bpm_diff = abs(float(t["bpm"]) - float(track1["bpm"]))
                    key_diff = abs(t.get("key_compatibility", 0) - track1.get("key_compatibility", 0))
                    if bpm_diff < 8 and key_diff < 0.2:
                        compatible_tracks.append(i)

            if compatible_tracks:
                track2_idx = random.choice(compatible_tracks)
                track1_pos = track_ids.index(track1["id"])
                track2_pos = track_ids.index(tracks[track2_idx]["id"])
                batch_X.append([track1_pos, track2_pos])
                batch_y.append(1.0)

                # Generate negative sample
                while True:
                    neg_idx = random.randint(0, num_tracks - 1)
                    if neg_idx not in compatible_tracks and neg_idx != track1_idx:
                        break

                neg_pos = track_ids.index(tracks[neg_idx]["id"])
                batch_X.append([track1_pos, neg_pos])
                batch_y.append(0.0)

        X.extend(batch_X)
        y.extend(batch_y)

        if len(X) >= num_samples:
            break

    return np.array(X, dtype=np.int64), np.array(y, dtype=np.float32)

def train_ncf_model(tracks: List[dict], epochs: int = 5, batch_size: int = 32,
                    model_path: str = "models/ncf_model.pth") -> NCF:
    """Train the NCF model with batch processing"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training NCF model on {device}")

    model = NCF(num_tracks=len(tracks)).to(device)

    # Load existing model if available
    if os.path.exists(model_path):
        logging.info("Loading existing model weights...")
        try:
            model.load_model(model_path)
            return model
        except Exception as e:
            logging.warning(f"Could not load existing model: {e}")

    try:
        # Generate training data
        X, y = generate_training_data(tracks, num_samples=min(10000, len(tracks) * 20))

        # Convert to PyTorch tensors
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(X),
            torch.FloatTensor(y)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X[:, 0], batch_X[:, 1]).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            logging.info(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

        # Save trained model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        logging.info(f"Model saved to {model_path}")

    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

    return model

def recommend_tracks(track_id: str, index: faiss.Index, track_vectors: np.ndarray,
                     track_ids: np.ndarray, model: NCF, tracks: List[dict],
                     n_recommendations: int = 5) -> List[dict]:
    """Get track recommendations using hybrid approach"""
    try:
        query_index = np.where(track_ids == track_id)[0][0]
    except IndexError:
        logging.error(f"Track ID {track_id} not found in dataset.")
        return []

    query_vector = np.array([track_vectors[query_index]])

    # FAISS Similarity Search
    D, I = index.search(query_vector, n_recommendations * 2)  # Get more candidates
    candidates = [track_ids[i] for i in I[0] if i != query_index]

    # NCF Refinement
    device = next(model.parameters()).device
    track_idx = track_ids.tolist().index(track_id)
    candidate_scores = []

    for candidate in candidates:
        candidate_idx = track_ids.tolist().index(candidate)
        with torch.no_grad():
            score = model(
                torch.LongTensor([track_idx]).to(device),
                torch.LongTensor([candidate_idx]).to(device)
            ).item()
        candidate_scores.append((candidate, score))

    # Sort by NCF score and take top N
    final_recommendations = sorted(candidate_scores, key=lambda x: x[1], reverse=True)[:n_recommendations]

    # Return full track information
    recommended_tracks = []
    for rec_id, score in final_recommendations:
        track = next(t for t in tracks if t["id"] == rec_id)
        track["recommendation_score"] = score
        recommended_tracks.append(track)

    return recommended_tracks

def print_track_info(track: dict, is_source: bool = False):
    """Pretty print track information"""
    prefix = "🎵 Source Track:" if is_source else "- Recommended:"
    print(f"\n{prefix} {track['title']} by {track['artist']}")
    print(f"  BPM: {track['bpm']}, Key: {track['key']}")
    if not is_source and "recommendation_score" in track:
        print(f"  Match Score: {track['recommendation_score']:.2f}")
    print(f"  Energy: {track['energy']:.2f}, Danceability: {track['danceability']:.2f}")

if __name__ == "__main__":
    # Load and process data
    REKORDBOX_FILE_PATH = os.path.join(os.getcwd(), "data", "rekordbox_tracks.xml")
    MODEL_PATH = os.path.join(os.getcwd(), "models", "ncf_model.pth")

    if not os.path.exists(REKORDBOX_FILE_PATH):
        logging.error("File not found at path")
        exit()

    tracks = fetch_all_tracks(REKORDBOX_FILE_PATH)
    if not tracks:
        logging.error("No tracks found")
        exit()

    # Build FAISS index
    index, track_vectors, track_ids = build_faiss_index(tracks)

    # Train or load NCF model
    model = train_ncf_model(tracks, model_path=MODEL_PATH)

    # Test recommendations
    test_track_id = track_ids[random.randint(0, len(track_ids) - 1)]
    test_track = next(t for t in tracks if t["id"] == test_track_id)

    recommended = recommend_tracks(
        test_track_id, index, track_vectors, track_ids, model, tracks
    )

    # Print results
    print("\n🎧 DJ Track Recommendations")
    print("=" * 50)
    print_track_info(test_track, is_source=True)

    print("\n🔊 Recommended Tracks:")
    for track in recommended:
        print_track_info(track)
