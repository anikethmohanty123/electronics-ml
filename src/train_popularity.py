import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from config import DATA_PROCESSED, MODEL_POPULARITY_PATH

def train_popularity_model():
    print(f"Loading processed data from: {DATA_PROCESSED}")
    df = pd.read_csv(DATA_PROCESSED)

    feature_cols_text = ["Product_Name"]
    feature_cols_num = ["Price", "Review_Count", "is_in_stock"]
    target_col = "is_popular"

    X = df[feature_cols_text + feature_cols_num]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    text_transformer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, "Product_Name"),
            ("num", numeric_transformer, feature_cols_num),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("Training popularity model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_POPULARITY_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_POPULARITY_PATH)
    print(f"Saved popularity model to: {MODEL_POPULARITY_PATH}")

if __name__ == "__main__":
    train_popularity_model()
