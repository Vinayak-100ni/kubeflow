import pandas as pd
import os

# Create processed folder
os.makedirs("data/processed", exist_ok=True)

# Load raw dataset
df = pd.read_csv("../data/Telco_Cusomer_Churn.csv")

# Remove missing values
df = df.dropna()

# Save processed data
df.to_csv("../data/processed/cleaned.csv", index=False)

print("Data preparation completed")