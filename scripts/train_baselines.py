# /src/03_train_baselines.py
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

INP = Path("data/processed/tokens_labeled.csv")


def make_preprocessor():
    # Text → TF-IDF
    curr_tok_vect = ("curr_tok_vect", TfidfVectorizer(min_df=3, ngram_range=(1,1)), "curr_tok")
    prev_tok_vect = ("prev_tok_vect", TfidfVectorizer(min_df=3, ngram_range=(1,1)), "prev_tok")

    # Categorical one-hot
    cat_cols = ["curr_pos", "prev_pos", "prev_lang", "speaker", "generation_strategy"]
    cat = ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)

    # Numeric standardized (sparse-safe: with_mean=False)
    num_cols = ["curr_len", "prev_len", "position_idx", "position_frac", "is_punct", "prev_is_punct"]
    num = ("num", StandardScaler(with_mean=False), num_cols)

    return ColumnTransformer(
        transformers=[curr_tok_vect, prev_tok_vect, cat, num],
        remainder="drop",
        verbose_feature_names_out=False
    )

def train_eval(X_train, y_train, X_test, y_test, model, name):
    pipe = Pipeline(steps=[("pre", make_preprocessor()), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=3))

def main():
    df = pd.read_csv(INP)
    df["curr_tok"] = df["curr_tok"].fillna("").astype(str)
    df["prev_tok"] = df["prev_tok"].fillna("").astype(str)

    # Optional: drop first token in each utterance (label is always 0 by construction)
    df = df[df["position_idx"] > 0].copy()

    features = [
        "curr_tok", "prev_tok", "curr_pos", "prev_pos", "prev_lang",
        "speaker", "generation_strategy",
        "curr_len", "prev_len", "position_idx", "position_frac", "is_punct", "prev_is_punct"
    ]
    X = df[features]
    y = df["y_switch"].astype(int)

    # Group-wise split by conversation_id to avoid leakage
    groups = df["conversation_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Logistic Regression baseline (balanced)
    logreg = LogisticRegression(max_iter=200, class_weight="balanced")
    train_eval(X_train, y_train, X_test, y_test, logreg, "Logistic Regression (balanced)")

    # Random Forest baseline (balanced)
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    train_eval(X_train, y_train, X_test, y_test, rf, "Random Forest (balanced_subsample)")

if __name__ == "__main__":
    main()
