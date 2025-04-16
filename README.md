# Hybrid job recommendation 

A personalized AI-powered job recommendation system using NLP, deep learning, and retrieval-based techniques. This project combines traditional and modern search techniques with large language models to suggest the most relevant job listings to users.

## Features

- *Text-based Job Search* using BM25 and TF-IDF
- *Semantic Search* with BERT embeddings + FAISS
- *Deep Learning Similarity Model* trained on user-job interactions
- *Hybrid Ranking* combining lexical, semantic, and DL signals
- *RAG with LLMs (OpenAI)* for personalized job suggestions
- *MongoDB Integration* to store and query job listings
- *LangChain Chatbot* to answer job-related queries

## Tech Stack

- *Python, **PyTorch, **Transformers*
- *FAISS, **Scikit-learn, **LangChain, **OpenAI API*
- *MongoDB, **Pandas, **NumPy*
- *Streamlit* or *Gradio* (for chatbot interface)
- *Docker/Kubernetes* (for deployment)

## Project Structure

src/ ├── data/                  # User-job interaction generator ├── models/                # Deep model, RAG, inference ├── pipelines/             # Retrieval and vectorization logic ├── utils.py               # Utility functions ├── database.py            # MongoDB connection artifacts/                 # Saved vectorizers, FAISS index, models notebook/                  # Exploratory data analysis, simulations

## How it Works

1. *Preprocess* job listings and generate embeddings
2. *Retrieve* jobs using BM25, FAISS, and/or deep model
3. *Rank* jobs with hybrid scoring strategy
4. *Recommend* top jobs or use LLM to explain results


# Ensure you set up:

.env with your OpenAI API key

MongoDB running with jobportal.jobdetails collection


# Run Pipelines

1.Train Model:

python src/pipelines/training_pipeline.py

2.Test Vectorization & Retrieval:

python src/pipelines/test_vectorization.py

3.Deep Inference:

python src/models/deep_inference.py

4.LangChain RAG:

python src/models/rag_chain.py


# Future Plans

Web UI with React.js

Deployment on AWS/GCP

API-based monetization model