# src/models/model.py

import torch
import torch.nn as nn

class SimilarityModel(nn.Module):
    def __init__(self, embedding_dim=384):
        super(SimilarityModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_emb, job_emb):
        # Concatenate user and job embeddings
        x = torch.cat((user_emb, job_emb), dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))  # Output similarity score (0-1)
        return x