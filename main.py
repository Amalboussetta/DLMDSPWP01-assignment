import pandas as pd
from sqlalchemy import create_engine

# === 1. Load CSV files ===
train_df = pd.read_csv("train.csv")   # 4 training functions
ideal_df = pd.read_csv("ideal.csv")   # 50 ideal functions

# Keep only x and the 4 y-columns for training table
training_cols = ["x", "y1", "y2", "y3", "y4"]
train_df = train_df[training_cols]

# === 2. Create SQLite DB connection ===
engine = create_engine("sqlite:///assignment.db", echo=False)

# === 3. Save training data ===
train_df.to_sql("training_data", con=engine, if_exists="replace", index=False)

# === 4. Save ideal functions data ===
ideal_df.to_sql("ideal_functions", con=engine, if_exists="replace", index=False)

print("✅ Training and ideal data loaded into SQLite database (assignment.db)")
