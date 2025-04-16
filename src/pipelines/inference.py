import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from src.models.deep_model import SimilarityModel
from src.pipelines.retrival import compute_dl_similarity

#  Load model and set to eval
model = SimilarityModel()
model.load_state_dict(torch.load("artifacts/similarity_model.pt"))
model.eval()

#  Load SentenceTransformer
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

#  Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
collection = client["jobportal"]["jobdetails"]
jobs = list(collection.find({}, {"_id": 0, "cleaned_text": 1}))
job_texts = [job["cleaned_text"] for job in jobs]

#  Get embeddings for all job descriptions (consider caching in future)
job_embeddings = bert_model.encode(job_texts, convert_to_tensor=True)

def get_top_jobs_deep_model(query, top_k=5):
    query_emb = bert_model.encode(query, convert_to_tensor=True).unsqueeze(0)

    scores = []
    with torch.no_grad():
        for job_emb in job_embeddings:
            score = model(query_emb, job_emb.unsqueeze(0)).item()
            scores.append(score)

    top_indices = np.argsort(scores)[::-1][:top_k]
    top_jobs = [(job_texts[i], scores[i]) for i in top_indices]

    return top_jobs

if __name__ == "__main__":
    user_query = input("Enter your job query: ")
    results = get_top_jobs_deep_model(user_query, top_k=5)
    
    print("\n🔍 Top Jobs by Deep Model Similarity:\n")
    for i, (text, score) in enumerate(results):
        print(f"{i+1}. Score: {score:.4f}")
        print(f"   Job: {text[:150]}...\n")
        
query = "data analyst with Python and SQL"
positive_example = "Looking for data analyst with experience in SQL and Python scripting"
negative_example = "Frontend developer skilled in React and UI/UX"

scores = compute_dl_similarity(query, [positive_example, negative_example])
print(scores)  # Expect higher score for the positive one

