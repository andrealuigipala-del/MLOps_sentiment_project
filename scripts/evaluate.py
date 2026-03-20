# scripts/evaluate.py

from transformers import pipeline
from sklearn.metrics import classification_report
from scripts.preprocess import load_dataset

# Mappatura label modello → numeri
label_map = {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
}

def evaluate_model(model_name, data_url):
    """
    Valuta un modello di sentiment su un dataset di test.

    Args:
        model_name (str): nome modello HuggingFace o path locale
        data_url (str): URL del dataset CSV
    """

    # 1. Carica dati
    _, X_test, _, y_test = load_dataset(data_url)

    # 2. Carica modello come pipeline
    classifier = pipeline("sentiment-analysis", model=model_name)

    # 3. Predizioni
    predictions = classifier(X_test.tolist(), batch_size=32)

    # 4. Converte label testuali → numeriche
    y_pred = [label_map[p["label"]] for p in predictions]

    # 5. Report
    print(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"]))
