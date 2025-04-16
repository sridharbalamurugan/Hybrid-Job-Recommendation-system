import os
import asyncio
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import Document, AIMessage
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_openai import OpenAIEmbeddings, OpenAI  #  Disabled due to no API access

from src.pipelines.vectorization import load_faiss_index
from src.pipelines.retrival import get_top_faiss_jobs

#  Disable Streamlit watcher bug (optional)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

#  Load FAISS index and job texts
faiss_index, all_job_texts = load_faiss_index(return_texts=True)

#  MOCK: Dummy embedding function (to prevent API call)
class DummyEmbeddings:
    def embed_query(self, text):
        return [0.1] * 384  # dummy vector of expected size

embedding = DummyEmbeddings()

# LangChain-compatible FAISS retriever (mocked embeddings)
retriever = FAISS(
    embedding_function=embedding,
    index=faiss_index,
    docstore=InMemoryDocstore({
        str(i): Document(page_content=text) for i, text in enumerate(all_job_texts)
    }),
    index_to_docstore_id={i: str(i) for i in range(len(all_job_texts))}
)

#  Prompt template
prompt_template = PromptTemplate(
    input_variables=["query", "job_context"],
    template="""
You are a helpful and knowledgeable AI career assistant.

A user is looking for jobs related to: "{query}"

Here are some relevant job descriptions:

{job_context}

Please suggest 3 specific jobs that best match the user's interests.
For each job, explain in 1–2 sentences why it is a good fit.
Use a friendly, natural tone in your response.
"""
)

# Mocked LLMChain-like function
def mock_llm_chain_run(inputs):
    query = inputs["query"]
    job_context = inputs["job_context"]

    print("\n Final Prompt Sent to LLM:\n")
    print(prompt_template.format(query=query, job_context=job_context))

    #  Dummy response
    return AIMessage(content="""
Here are 3 job suggestions just for you:

1. **NLP Engineer at BrainTech** – Great for your NLP background, this role focuses on building real-time chatbots using large language models.

2. **Remote ML Researcher at DeepAI Labs** – A fully remote position where you can explore cutting-edge research in natural language understanding.

3. **AI Consultant at DataBoost** – Combines your technical skills and client interaction for real-world NLP applications in healthcare and finance.
""")

#  Truncate job descriptions
def truncate_text(text, limit=300):
    return text[:limit].rsplit('.', 1)[0] + '.' if '.' in text[:limit] else text[:limit] + '...'

#  Main RAG function
def generate_llm_recommendations(query: str, top_k: int = 5):
    top_jobs = get_top_faiss_jobs(query, top_k)
    top_indices = [int(idx) for _, idx in top_jobs]

    job_context = "\n\n".join([
        f"{i+1}. {truncate_text(all_job_texts[idx])}" for i, idx in enumerate(top_indices)
    ])

    #  Return mocked response
    return mock_llm_chain_run({
        "query": query,
        "job_context": job_context
    })

#  CLI safe main block
if __name__ == "__main__":
    try:
        query = input("Enter your job interest: ")
        print("\n AI-Powered Job Suggestions:\n")
        suggestions = generate_llm_recommendations(query)
        print(suggestions.content)
    except RuntimeError:
        # Fix event loop bug
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(asyncio.sleep(0.1))
        query = input("Enter your job interest: ")
        print("\n AI-Powered Job Suggestions:\n")
        suggestions = generate_llm_recommendations(query)
        print(suggestions.content)
