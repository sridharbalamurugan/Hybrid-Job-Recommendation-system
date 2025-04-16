import faiss
import numpy as np
import pickle
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

#  Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["jobportal"]
collection = db["jobdetails"]

#  Fetch cleaned job descriptions from MongoDB
cursor = collection.find({}, {"_id": 0, "cleaned_text": 1})
cleaned_data = [doc["cleaned_text"] for doc in cursor]

#  Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_data)

#  Compute BM25
tokenized_docs = [doc.lower().split() for doc in cleaned_data]
bm25 = BM25Okapi(tokenized_docs)

# Load pre-trained BERT-based model
model = SentenceTransformer("all-MiniLM-L6-v2")

#  Generate BERT-based embeddings
job_embeddings = model.encode(cleaned_data)

#  Store embeddings in FAISS
dimension = job_embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW index with 32 neighbors
index.hnsw.efConstruction = 64  # Adjust for recall-performance balance
index.hnsw.efSearch = 32  # Search complexity tuning
normalized_embeddings = job_embeddings / np.linalg.norm(job_embeddings, axis=1, keepdims=True)
index.add(np.array(normalized_embeddings))


#  Save vectorizer, TF-IDF matrix, BM25, and FAISS index
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(tfidf_vectorizer, vec_file)

with open("job_vectors.pkl", "wb") as vec_file:
    pickle.dump(tfidf_matrix, vec_file)

with open("bm25.pkl", "wb") as bm25_file:
    pickle.dump(bm25, bm25_file)

faiss.write_index(index, "faiss_index.idx")
print(f" Successfully processed {len(cleaned_data)} job descriptions!")

def load_faiss_index(return_texts=False):
    index = faiss.read_index("artifacts/faiss_index.idx")

    if return_texts:
        with open("artifacts/job_texts.pkl", "rb") as f:
            job_texts = pickle.load(f)
        return index, job_texts

    return index

with open("artifacts/job_texts.pkl", "wb") as f:
    pickle.dump(cleaned_data, f)

