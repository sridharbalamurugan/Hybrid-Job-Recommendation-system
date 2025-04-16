import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("notebook/user_job_pairs.csv")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("notebook/user_job_train.csv", index=False)
val_df.to_csv("notebook/user_job_val.csv", index=False)

print(" Split into train and val sets.")
