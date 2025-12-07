import joblib
from config import MODEL_RATING_PATH, MODEL_POPULARITY_PATH
import pandas as pd


def load_models():
    rating_model = joblib.load(MODEL_RATING_PATH)
    popularity_model = joblib.load(MODEL_POPULARITY_PATH)
    return rating_model, popularity_model

def predict_for_product(product_name: str, price: float, review_count: int, availability: str):
    rating_model, popularity_model = load_models()

    is_in_stock = 1 if "in stock" in availability.lower() else 0

    sample = pd.DataFrame({
    "Product_Name": [product_name],
    "Price": [price],
    "Review_Count": [review_count],
    "is_in_stock": [is_in_stock],
})

    predicted_rating = rating_model.predict(sample)[0]
    predicted_popularity = popularity_model.predict(sample)[0]
    predicted_popularity_prob = popularity_model.predict_proba(sample)[0]

    return predicted_rating, predicted_popularity, predicted_popularity_prob

if __name__ == "__main__":
    print("=== Electronics Product Predictor ===")
    name = input("Product name: ")
    price = float(input("Price: "))
    rc = int(input("Review count: "))
    avail = input("Availability (In Stock / Out of Stock): ")

    rating, pop, probs = predict_for_product(name, price, rc, avail)

    print(f"\nPredicted rating: {rating:.2f}")
    print(f"Popularity label (1=popular): {pop}")
    print(f"Probability distribution: {probs}")
