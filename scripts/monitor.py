# scripts/monitor.py
import json
import os
import random
import pandas as pd
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

LOG_PATH    = "./data/monitoring/metrics_log.json"
MODEL_DIR   = "./data/results/trained_model"
DATA_URL    = "https://raw.githubusercontent.com/andrealuigipala-del/MLOps_sentiment_project/refs/heads/main/data/src/Twitter_Data.csv"
F1_THRESHOLD = 0.7
SAMPLE_SIZE  = 150


def load_log():
    """Carica il log esistente o restituisce lista vuota."""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return []


def save_log(log):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def evaluate_sample(model_dir, texts, labels, batch_size=16):
    """Valuta il modello su una lista di testi e label."""
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=64, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            all_preds.extend(torch.argmax(logits, dim=1).tolist())

    f1 = f1_score(labels, all_preds, average="macro")
    return round(f1, 4), all_preds


def get_sample_seed(log):
    """
    Seed diverso ad ogni run basato sul numero di run già eseguite.
    Simula la variazione che si avrebbe con dati nuovi in produzione.
    """
    return len(log) * 42


def main():
    # 1. Carica log esistente
    log = load_log()
    run_number = len(log) + 1
    print(f"\n=== MONITOR — Run #{run_number} ===")

    # 2. Carica dataset
    df = pd.read_csv(DATA_URL)
    df = df.dropna(subset=['clean_text', 'category'])
    df['clean_text'] = df['clean_text'].astype(str)
    df['category']   = df['category'] + 1          # shift coerente con preprocess.py
    df['category']   = df['category'].astype(int)

    # 3. Campione casuale con seed diverso ad ogni run
    seed = get_sample_seed(log)
    sample = df.sample(SAMPLE_SIZE, random_state=seed)
    texts  = sample['clean_text'].tolist()
    labels = sample['category'].tolist()

    print(f"Campione: {SAMPLE_SIZE} esempi, seed={seed}")

    # 4. Valutazione
    print("Contenuto model dir:", os.listdir(MODEL_DIR))
    f1, preds = evaluate_sample(MODEL_DIR, texts, labels)
    print(f"F1 macro: {f1}")

    # 5. Distribuzione predizioni (utile per rilevare drift)
    dist = {
        "negative": preds.count(0),
        "neutral":  preds.count(1),
        "positive": preds.count(2)
    }
    print(f"Distribuzione predizioni: {dist}")

    # 6. Salva entry nel log
    entry = {
        "run":          run_number,
        "timestamp":    datetime.utcnow().isoformat(),
        "f1_macro":     f1,
        "sample_seed":  seed,
        "sample_size":  SAMPLE_SIZE,
        "pred_dist":    dist,
        "alert":        f1 < F1_THRESHOLD
    }
    log.append(entry)
    save_log(log)
    print(f"Log aggiornato: {LOG_PATH}")

    # 7. Alert se F1 sotto soglia
    if f1 < F1_THRESHOLD:
        raise ValueError(
            f"[ALERT] F1={f1} sotto la soglia {F1_THRESHOLD}. "
            f"Retraining con più dati necessario."
        )

    print("Monitoraggio completato con successo.\n")


if __name__ == "__main__":
    main()
