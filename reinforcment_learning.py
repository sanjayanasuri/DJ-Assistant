import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultiArmedBandit:
    def __init__(self, num_tracks, alpha=0.1, gamma=0.9):
        """
        Implements a Multi-Armed Bandit using Thompson Sampling for track recommendations.
        :param num_tracks: Number of unique tracks
        :param alpha: Learning rate for updates
        :param gamma: Discount factor for rewards
        """
        self.num_tracks = num_tracks
        self.alpha = alpha
        self.gamma = gamma
        self.track_counts = np.zeros(num_tracks)
        self.track_rewards = np.zeros(num_tracks)

    def select_track(self):
        """
        Select a track based on Thompson Sampling.
        """
        sampled_values = np.random.beta(self.track_rewards + 1, self.track_counts - self.track_rewards + 1)
        selected_track = np.argmax(sampled_values)
        return selected_track

    def update_rewards(self, track_id, reward):
        """
        Update the reward estimates for a given track.
        :param track_id: ID of the track being updated
        :param reward: Reward value (1 for positive feedback, 0 for negative feedback)
        """
        self.track_counts[track_id] += 1
        self.track_rewards[track_id] += self.alpha * (reward - self.track_rewards[track_id])
        logging.info(f"Updated track {track_id} with reward {reward}")

if __name__ == "__main__":
    num_tracks = 50
    bandit = MultiArmedBandit(num_tracks)
    
    for _ in range(10):  # Simulate 10 feedback rounds
        track_id = bandit.select_track()
        reward = random.choice([0, 1])  # Simulated user feedback
        bandit.update_rewards(track_id, reward)
        logging.info(f"Track {track_id} selected, received reward {reward}")
