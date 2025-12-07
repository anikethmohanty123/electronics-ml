from src.preprocess import preprocess
from src.train_rating import train_rating_model
from src.train_popularity import train_popularity_model

if __name__ == "__main__":
    print("STEP 1 — Preprocessing...")
    preprocess()

    print("\nSTEP 2 — Training rating model...")
    train_rating_model()

    print("\nSTEP 3 — Training popularity model...")
    train_popularity_model()

    print("\nDONE! Models are trained and saved.")
