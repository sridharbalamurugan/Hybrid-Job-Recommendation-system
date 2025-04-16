from pymongo import MongoClient
import random
import pandas as pd

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
collection = client["jobportal"]["jobdetails"]
jobs = list(collection.find({}, {"_id": 0, "cleaned_text": 1}))

# Simulated user profiles
user_profiles = [
    "machine learning python data analysis",
    "frontend react javascript html css",
    "cloud computing devops aws azure docker",
    "digital marketing seo content strategy",
    "ui ux design figma adobe creative"
]

# Generate interaction pairs
interaction_data = []

for user_text in user_profiles:
    relevant_jobs = random.sample(jobs, 3)  # simulate top 3 relevant
    irrelevant_jobs = random.sample(jobs, 3)  # simulate 3 random irrelevant

    for job in relevant_jobs:
        interaction_data.append({
            "user": user_text,
            "job": job["cleaned_text"],
            "label": 1
        })

    for job in irrelevant_jobs:
        interaction_data.append({
            "user": user_text,
            "job": job["cleaned_text"],
            "label": 0
        })

# Save to CSV (for PyTorch DataLoader later)
df = pd.DataFrame(interaction_data)
df.to_csv("notebook/user_job_pairs.csv", index=False)
print(" Interaction pairs saved to notebook/user_job_pairs.csv")