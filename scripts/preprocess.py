# scripts/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(url, test_size=0.2, random_state=42):
    """
    Carica un dataset CSV da un URL pubblico, divide in feature e target, 
    e restituisce train e test split.

    Args:
        url (str): link al file CSV pubblico
        test_size (float): proporzione dei dati da usare come test
        random_state (int): seed per riproducibilità

    Returns:
        X_train, X_test, y_train, y_test (pd.Series o pd.DataFrame)
    """
    # Carica il CSV dall'URL
    df = pd.read_csv(url)

    # Controlla che ci siano le colonne attese
    if 'clean_text' not in df.columns or 'category' not in df.columns:
        raise ValueError("Il CSV deve contenere le colonne 'text' e 'category'")

    # Seleziona feature e target
    X = df['clean_text']
    y = df['category']

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
