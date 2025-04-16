import torch
import warnings
import streamlit as st
from src.LLM.rag_chain import generate_llm_recommendations


warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")
st.set_page_config(page_title="AI Job Assistant", layout="wide")
st.title("💼 AI-Powered Job Assistant")

query = st.text_input("Ask something about jobs:", placeholder="e.g. Show me data science jobs with remote option")

if query:
    with st.spinner("Fetching recommendations..."):
        response = generate_llm_recommendations(query)
        st.markdown("### 🤖 Recommendations")
        st.write(response)
