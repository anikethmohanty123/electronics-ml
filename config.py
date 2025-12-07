import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "amazon_all_electronics_data.csv")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "electronics_processed.csv")

MODEL_RATING_PATH = os.path.join(BASE_DIR, "models", "rating_model.joblib")
MODEL_POPULARITY_PATH = os.path.join(BASE_DIR, "models", "popularity_model.joblib")
