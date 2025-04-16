import pandas as pd
from pymongo import MongoClient  # type: ignore

# Load the cleaned CSV file
df_cleaned = pd.read_csv(r"C:\Users\DELL\jobportalrecommendationsystem\notebook\cleaned_job_data.csv")

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["jobportal"]  # Database name
collection = db["jobdetails"]  # Collection name

# ✅ Step 1: Clear existing data
collection.delete_many({})  # Remove all existing documents

# ✅ Step 2: Insert new cleaned data
data_dict = df_cleaned.to_dict("records")
collection.insert_many(data_dict)

print(f"✅ Successfully replaced with {len(data_dict)} new cleaned records!")

# ✅ Optional: Verify by printing one document
print(collection.find_one())
