import os
import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_PROCESSED

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")

def ensure_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Distribution of Prices
def plot_price_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df["Price"], bins=40, color="skyblue", edgecolor="black")
    plt.title("Price Distribution of Electronics")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "price_distribution.png"))
    plt.close()

# 2. Distribution of Ratings
def plot_rating_distribution(df):
    plt.figure(figsize=(8, 6))
    plt.hist(df["Rating"], bins=[1,2,3,4,5,6], color="lightgreen", edgecolor="black")
    plt.title("Rating Distribution")
    plt.xlabel("Rating (1–5)")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "rating_distribution.png"))
    plt.close()

# 3. Review Count Distribution (log scale)
def plot_review_count_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df["Review_Count"], bins=40, color="orange", edgecolor="black")
    plt.xscale("log")
    plt.title("Review Count Distribution (Log Scale)")
    plt.xlabel("Review Count (log scale)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "review_count_distribution.png"))
    plt.close()

# 4. Price vs Rating Scatter Plot
def plot_price_vs_rating(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Price"], df["Rating"], alpha=0.4, color="purple")
    plt.title("Price vs Rating")
    plt.xlabel("Price")
    plt.ylabel("Rating")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "price_vs_rating.png"))
    plt.close()

# 5. Popular vs Not Popular Bar Chart
def plot_popularity_counts(df):
    plt.figure(figsize=(8, 6))
    counts = df["is_popular"].value_counts()
    plt.bar(["Not Popular", "Popular"], counts, color=["gray", "gold"])
    plt.title("Popularity Breakdown")
    plt.ylabel("Number of Products")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "popularity_breakdown.png"))
    plt.close()

def generate_all_visualizations():
    ensure_dir()
    df = pd.read_csv(DATA_PROCESSED)

    print(f"Loaded processed data: {df.shape}")
    print("Generating visualizations...")

    plot_price_distribution(df)
    plot_rating_distribution(df)
    plot_review_count_distribution(df)
    plot_price_vs_rating(df)
    plot_popularity_counts(df)

    print("Visualizations saved to:", SAVE_DIR)

if __name__ == "__main__":
    generate_all_visualizations()
