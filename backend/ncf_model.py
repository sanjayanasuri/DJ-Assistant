import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dim=64):
        """
        Neural Collaborative Filtering Model
        :param num_users: Number of unique users
        :param num_items: Number of unique items (tracks)
        :param embedding_dim: Dimension of embedding layers
        :param hidden_dim: Dimension of hidden layers
        """
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        x = torch.cat([user_embed, item_embed], dim=-1)
        return self.fc_layers(x)

def train_ncf(model, train_loader, epochs=10, lr=0.001):
    """
    Train the NCF model
    :param model: NCF model instance
    :param train_loader: DataLoader for training
    :param epochs: Number of training epochs
    :param lr: Learning rate
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for users, items, labels in train_loader:
            optimizer.zero_grad()
            preds = model(users, items).squeeze()
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

def create_dataloader(user_item_df, batch_size=128):
    """
    Create DataLoader from user-item interaction data
    :param user_item_df: Pandas DataFrame with columns [user_id, item_id, label]
    :param batch_size: Size of batches for training
    """
    user_tensor = torch.tensor(user_item_df['user_id'].values, dtype=torch.long)
    item_tensor = torch.tensor(user_item_df['item_id'].values, dtype=torch.long)
    label_tensor = torch.tensor(user_item_df['label'].values, dtype=torch.float32)
    dataset = TensorDataset(user_tensor, item_tensor, label_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # Example dataset
    num_users, num_items = 100, 500
    interactions = pd.DataFrame({
        'user_id': np.random.randint(0, num_users, 10000),
        'item_id': np.random.randint(0, num_items, 10000),
        'label': np.random.randint(0, 2, 10000)
    })
    
    # Prepare DataLoader
    train_loader = create_dataloader(interactions)
    
    # Initialize and train model
    model = NCF(num_users, num_items)
    train_ncf(model, train_loader)
