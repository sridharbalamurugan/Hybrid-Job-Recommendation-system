from src.pipelines.retrival import retrieve_hybrid, retrieve_bm25, retrieve_faiss, compute_dl_similarity
from pymongo import MongoClient

#  Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
collection = client["jobportal"]["jobdetails"]
all_jobs = list(collection.find({}, {"_id": 0, "cleaned_text": 1}))
all_job_texts = [job["cleaned_text"] for job in all_jobs]

#  User query
query = "machine learning data analysis python"
top_k = 5

print("\n Hybrid Ranking with DL Model Integration")

# Retrieve BM25 and FAISS results
bm25_indices, bm25_scores = retrieve_bm25(query, top_k=top_k)
faiss_indices, faiss_scores = retrieve_faiss(query, top_k=top_k)

# Convert indices to lists (in case they're numpy arrays)
bm25_indices = list(bm25_indices)
faiss_indices = list(faiss_indices)

#  Get union of all indices from BM25 and FAISS
retrieved_indices = list(set(bm25_indices + faiss_indices))
valid_retrieved_indices = [i for i in retrieved_indices if i < len(all_job_texts)]

#  Prepare job texts and score lists
top_job_texts = [all_job_texts[i] for i in valid_retrieved_indices]

# Match scores to their indices or assign 0 if not found
bm25_scores = [bm25_scores[bm25_indices.index(i)] if i in bm25_indices else 0.0 for i in valid_retrieved_indices]
faiss_scores = [faiss_scores[faiss_indices.index(i)] if i in faiss_indices else 0.0 for i in valid_retrieved_indices]

#  Compute deep learning model similarity
dl_scores = compute_dl_similarity(query, top_job_texts)

#  Hybrid score: weighted combination
final_scores = [
    0.2* bm25 + 0.3* faiss + 0.5 * dl
    for bm25, faiss, dl in zip(bm25_scores, faiss_scores, dl_scores)
]

#  Sort by final score (descending)
sorted_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)

#  Show top-k results
for idx in sorted_indices[:top_k]:
    print(f"Hybrid Score: {final_scores[idx]:.4f} | BM25: {bm25_scores[idx]:.4f} | FAISS: {faiss_scores[idx]:.4f} | DL: {dl_scores[idx]:.4f}")
    print(f"Text: {top_job_texts[idx][:100]}...\n")
