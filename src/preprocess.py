import os
import pandas as pd
from config import DATA_RAW, DATA_PROCESSED

def preprocess():
    print(f"Loading raw data from: {DATA_RAW}")
    df = pd.read_csv(DATA_RAW)

    expected_cols = [
        "Product_Name", "Price", "Rating",
        "Review_Count", "ASIN", "Product_URL", "Availability"
    ]
    df = df[expected_cols].copy()

    df = df.dropna(subset=["Product_Name", "Price", "Rating", "Review_Count"])
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Review_Count"] = pd.to_numeric(df["Review_Count"], errors="coerce")
    df = df.dropna(subset=["Price", "Rating", "Review_Count"])

    df["Availability"] = df["Availability"].astype(str)
    df["is_in_stock"] = df["Availability"].str.contains("in stock", case=False).astype(int)

    popularity_threshold = df["Review_Count"].quantile(0.75)
    df["is_popular"] = (df["Review_Count"] >= popularity_threshold).astype(int)

    os.makedirs(os.path.dirname(DATA_PROCESSED), exist_ok=True)
    df.to_csv(DATA_PROCESSED, index=False)

    print(f"Saved processed data to: {DATA_PROCESSED}")
    print("Processed shape:", df.shape)
    print("Popularity threshold (Review_Count):", popularity_threshold)

if __name__ == "__main__":
    preprocess()
