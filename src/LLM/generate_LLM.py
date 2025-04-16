import openai
import os
import argparse

# Set your OpenAI key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

#  Format the prompt for real API or mock
def format_prompt(user_query, job_descriptions):
    prompt = f"""You are a job recommendation assistant.

User's query: "{user_query}"

Based on the following job descriptions, recommend the top 3 most relevant jobs and explain briefly why they fit:

"""
    for i, desc in enumerate(job_descriptions, 1):
        prompt += f"\nJob {i}:\n{desc[:500]}..."  # Truncate for brevity

    prompt += "\n\nReturn your recommendations in a numbered list with explanations."
    return prompt

#  Real LLM call

def generate_real_llm_recommendations(user_query, job_descriptions, model="gpt-3.5-turbo"):
    prompt = format_prompt(user_query, job_descriptions)

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an AI job match expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=700
    )

    return response['choices'][0]['message']['content']

#  Mock output (safe, no billing)
def generate_mock_llm_recommendations(user_query, job_descriptions):
    print("[Mock Mode] Skipping OpenAI API call.")
    return f"""
Here are 3 job suggestions just for you:

1. **Data Scientist at TechCorp** – Perfect for your background in Python and machine learning, especially with a focus on analytics.

2. **AI Analyst at InsightAI** – A great fit if you're interested in research-heavy data modeling and ML pipelines.

3. **ML Engineer at BuildML** – Combines your coding skills and interest in building production-grade models using Python and data analysis.
"""

#  Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate job recommendations using LLM.")
    parser.add_argument("--mode", type=str, default="mock", choices=["mock", "real"], help="Choose LLM mode: mock or real")
    args = parser.parse_args()

    # Sample input (replace later with actual hybrid model results)
    query = "python data analysis machine learning"
    jobs = [
        "Python data analyst with machine learning experience...",
        "Backend developer with some ML and data visualization skills...",
        "Research assistant in data science department...",
    ]

    print(f"\n User Query: {query}\n")

    if args.mode == "real":
        if not openai.api_key:
            raise ValueError(" OPENAI_API_KEY is missing in environment.")
        response = generate_real_llm_recommendations(query, jobs)
    else:
        response = generate_mock_llm_recommendations(query, jobs)

    print("\n LLM Suggestions:\n")
    print(response)
