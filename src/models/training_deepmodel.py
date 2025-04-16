# src/models/training_deepmodel.py

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from sklearn.metrics import roc_auc_score, f1_score
from src.models.deep_model import SimilarityModel
from src.models.dataset import UserJobDataset

#  Settings
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Load full dataset
dataset = UserJobDataset(csv_path="notebook/user_job_pairs.csv")

#  Split dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

#  Initialize model
model = SimilarityModel()
model = model.to(DEVICE)

#  Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#  Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for user_emb, job_emb, label in train_loader:
        user_emb = user_emb.to(DEVICE)
        job_emb = job_emb.to(DEVICE)
        label = label.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        output = model(user_emb, job_emb)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    #  Validation phase
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for user_emb, job_emb, label in val_loader:
            user_emb = user_emb.to(DEVICE)
            job_emb = job_emb.to(DEVICE)
            label = label.to(DEVICE).unsqueeze(1)

            output = model(user_emb, job_emb)
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(output.cpu().numpy())
            all_preds.extend((output > 0.5).int().cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)

    print(f" Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val AUC: {auc:.4f} | F1: {f1:.4f}")

#  Save model
torch.save(model.state_dict(), "artifacts/similarity_model.pt")
print("✅ Model saved to artifacts/similarity_model.pt")
