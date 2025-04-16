import faiss
import torch
import numpy as np
import pickle
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.deep_model import SimilarityModel
from rank_bm25 import BM25Okapi

#  Corrected Paths (Now using artifacts/)
VECTOR_PATH = "artifacts/"

#  Load pre-trained vectorizers and index
with open(VECTOR_PATH + "vectorizer.pkl", "rb") as vec_file:
    tfidf_vectorizer: TfidfVectorizer = pickle.load(vec_file)

with open(VECTOR_PATH + "job_vectors.pkl", "rb") as vec_file:
    tfidf_matrix = pickle.load(vec_file)

with open(VECTOR_PATH + "bm25.pkl", "rb") as bm25_file:
    bm25: BM25Okapi = pickle.load(bm25_file)

faiss_index = faiss.read_index(VECTOR_PATH + "faiss_index.idx")

#  Load pre-trained BERT model
model = SimilarityModel()
model.load_state_dict(torch.load("artifacts/similarity_model.pt"))
model.eval()

# Load BERT
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_dl_similarity(user_text, job_texts):
    user_emb = bert_model.encode(user_text, convert_to_tensor=True).unsqueeze(0)
    job_embs = bert_model.encode(job_texts, convert_to_tensor=True)

    scores = []
    with torch.no_grad():
        for job_emb in job_embs:
            score = model(user_emb, job_emb.unsqueeze(0)).item()
            scores.append(score)
    return scores
#  Retrieval functions
def retrieve_tfidf(query, top_k=10):
    query_vec = tfidf_vectorizer.transform([query])
    scores = np.dot(tfidf_matrix, query_vec.T).toarray().flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]


def retrieve_bm25(query, top_k=10):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]


def retrieve_faiss(query, top_k=10):
    query_embedding = bert_model.encode([query])
    D, I = faiss_index.search(np.array(query_embedding), top_k)

    #  Normalize FAISS distances (lower distance = higher similarity)
    D = np.array(D[0])
    semantic_scores = 1 - (D / (np.max(D) + 1e-5))  # Avoid division errors

    return I[0], semantic_scores


# Hybrid Ranking (BM25 + FAISS)
def retrieve_hybrid(query, top_k=20, alpha=0.9):
    """
    Combines BM25 and FAISS scores using a weighted sum.
    alpha: weight for BM25 (0.0 = only FAISS, 1.0 = only BM25)
    """
    # Retrieve individual results
    bm25_indices, bm25_scores = retrieve_bm25(query, top_k)
    faiss_indices, faiss_scores = retrieve_faiss(query, top_k)

    # Normalize BM25 scores (Min-Max scaling)
    bm25_scores = np.array(bm25_scores)
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    # Compute final hybrid scores
    final_scores = alpha * bm25_scores + (1 - alpha) * faiss_scores

    # Rank results
    sorted_indices = np.argsort(final_scores)[::-1]  # Sort descending
    ranked_jobs = [bm25_indices[i] for i in sorted_indices]  # Use BM25 indices

    return ranked_jobs, final_scores[sorted_indices]

# Connect to MongoDB (or load job descriptions)
client = MongoClient("mongodb://localhost:27017/")
collection = client["jobportal"]["jobdetails"]
jobs = list(collection.find({}, {"_id": 0, "cleaned_text": 1}))
job_texts = [job["cleaned_text"] for job in jobs]

# Wrapper function for FAISS retrieval with actual job texts
def get_top_faiss_jobs(query, top_k=10):
    indices, scores = retrieve_faiss(query, top_k)
    top_jobs = [(job_texts[i], scores[idx]) for idx, i in enumerate(indices)]
    return top_jobs

