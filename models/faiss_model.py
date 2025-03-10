import faiss
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FAISSSimilarity:
    def __init__(self, feature_dim: int):
        """Initialize FAISS index for fast similarity search."""
        self.index = faiss.IndexFlatL2(feature_dim)
        self.track_ids = []
        logging.info("Initialized FAISS index with L2 distance.")
    
    def build_index(self, feature_matrix: np.ndarray, track_ids: list):
        """Build FAISS index with given track features."""
        if not isinstance(feature_matrix, np.ndarray):
            raise ValueError("Feature matrix must be a NumPy array")
        
        self.track_ids = track_ids
        self.index.add(feature_matrix.astype(np.float32))
        logging.info(f"FAISS index built with {len(track_ids)} tracks.")
    
    def search_similar_tracks(self, query_vector: np.ndarray, k: int = 5):
        """Find k most similar tracks to the given query vector."""
        distances, indices = self.index.search(query_vector.astype(np.float32), k)
        
        results = []
        for i in range(len(indices[0])):
            if indices[0][i] < len(self.track_ids):
                results.append((self.track_ids[indices[0][i]], distances[0][i]))
        
        return results

if __name__ == "__main__":
    # Example usage
    sample_features = np.random.rand(100, 5).astype(np.float32)  # 100 tracks, 5D feature vector
    track_ids = [f"track_{i}" for i in range(100)]
    
    faiss_model = FAISSSimilarity(feature_dim=5)
    faiss_model.build_index(sample_features, track_ids)
    
    query = np.random.rand(1, 5).astype(np.float32)
    similar_tracks = faiss_model.search_similar_tracks(query, k=3)
    
    print("Similar tracks:", similar_tracks)
