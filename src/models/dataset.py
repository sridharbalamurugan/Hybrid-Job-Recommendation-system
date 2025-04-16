import torch
from torch.utils.data import Dataset
import pandas as pd
from sentence_transformers import SentenceTransformer

class UserJobDataset(Dataset):
    def __init__(self, csv_path, model_name="all-MiniLM-L6-v2"):
        self.data = pd.read_csv(csv_path)
        self.model = SentenceTransformer(model_name)

        # Optional: Precompute all embeddings (recommended if data isn't huge)
        self.user_embeddings = self.model.encode(self.data["user"].tolist(), convert_to_tensor=True)
        self.job_embeddings = self.model.encode(self.data["job"].tolist(), convert_to_tensor=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_emb = self.user_embeddings[idx]
        job_emb = self.job_embeddings[idx]
        label = torch.tensor(self.data.iloc[idx]["label"], dtype=torch.float)
        return user_emb, job_emb, label
