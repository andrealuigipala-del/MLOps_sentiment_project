# scripts/evaluate_and_save.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import random
from huggingface_hub import snapshot_download
import os

HF_USERNAME = "andrealuigipala"
HF_REPO = "trained-model"  # nome fisso sul hub, NON cambiare

# Funzione di valutazione
def evaluate_model(model_dir, X, y, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    all_preds = []
    for i in range(0, len(X), batch_size):
        batch_texts = X[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).tolist()
            all_preds.extend(batch_preds)

    f1_macro = f1_score(y, all_preds, average="macro")
    report = classification_report(y, all_preds, digits=4)
    return f1_macro, report

def main():
    local_model_dir = "./data/results/final_model"

    # Carica dataset di test
    DATA_PATH = "https://raw.githubusercontent.com/andrealuigipala-del/MLOps_Final_Project/refs/heads/main/Twitter_Data.csv"
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['category', 'clean_text'])
    X_test = df['clean_text'].fillna("").astype(str)
    y_test = df['category'].astype(int)

    # Riduzione campioni per valutazione
    sample_size = 100
    indices = random.sample(range(len(X_test)), sample_size)
    X_test_small = X_test.iloc[indices].tolist()
    y_test_small = y_test.iloc[indices].tolist()

    # --- Valutazione del nuovo modello locale ---
    print(f"Valutando il modello locale: {local_model_dir}")
    new_f1, new_report = evaluate_model(local_model_dir, X_test_small, y_test_small)
    print("Nuovo modello F1 macro:", new_f1)

    # --- Controlla modello precedente su Hugging Face ---
    old_f1 = -1
    hub_dir = "./hf_models"
    os.makedirs(hub_dir, exist_ok=True)
    try:
        snapshot_path = snapshot_download(f"{HF_USERNAME}/{HF_REPO}", local_dir=hub_dir, local_dir_use_symlinks=False)
        print(f"Trovato modello precedente su Hugging Face: {HF_REPO}")
        old_f1, _ = evaluate_model(snapshot_path, X_test_small, y_test_small)
        print("Vecchio modello F1 macro:", old_f1)
    except Exception:
        print("Nessun modello precedente trovato su Hugging Face.")

    # --- Confronto e push ---
    if new_f1 >= old_f1:
        print("Il nuovo modello è migliore o pari al precedente. Sovrascriviamo su Hugging Face...")
        model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        model.push_to_hub(f"{HF_USERNAME}/{HF_REPO}", use_auth_token=True)
        tokenizer.push_to_hub(f"{HF_USERNAME}/{HF_REPO}", use_auth_token=True)
        print(f"Modello pushato su Hugging Face: {HF_REPO}")
    else:
        print("Il modello precedente è migliore. Nessun push del nuovo modello.")

    print("\nReport dettagliato del nuovo modello:")
    print(new_report)

if __name__ == "__main__":
    main()
