# scripts/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(url, test_size=0.2, random_state=42):
    """
    Carica un dataset CSV, applica lo shift +1 alle label
    (da -1/0/1 a 0/1/2) e restituisce train/test split.
    """
    df = pd.read_csv(url)

    if 'clean_text' not in df.columns or 'category' not in df.columns:
        raise ValueError("Il CSV deve contenere 'clean_text' e 'category'")

    df = df.dropna(subset=['clean_text', 'category'])
    df['clean_text'] = df['clean_text'].astype(str)
    df['category'] = df['category'] + 1        # shift: -1→0, 0→1, 1→2
    df['category'] = df['category'].astype(int)

    X = df['clean_text']
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Dataset caricato: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test
