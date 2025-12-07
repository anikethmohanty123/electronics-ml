import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from config import DATA_PROCESSED, MODEL_RATING_PATH

def train_rating_model():
    print(f"Loading processed data from: {DATA_PROCESSED}")
    df = pd.read_csv(DATA_PROCESSED)

    feature_cols_text = ["Product_Name"]
    feature_cols_num = ["Price", "Review_Count", "is_in_stock"]
    target_col = "Rating"

    X = df[feature_cols_text + feature_cols_num]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    text_transformer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, "Product_Name"),
            ("num", numeric_transformer, feature_cols_num),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("Training rating prediction model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Rating prediction R^2: {r2:.4f}")
    print(f"Rating prediction MAE: {mae:.4f}")

    os.makedirs(os.path.dirname(MODEL_RATING_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_RATING_PATH)
    print(f"Saved rating model to: {MODEL_RATING_PATH}")

if __name__ == "__main__":
    train_rating_model()
