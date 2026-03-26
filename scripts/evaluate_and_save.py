# scripts/evaluate_and_save.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import random
from huggingface_hub import snapshot_download
from datetime import datetime
import os

HF_USERNAME = "andrealuigipala"  # tuo username su HF
HF_REPO = "trained-model"        # base del repo

# Funzione di valutazione
def evaluate_model(model_dir, X, y, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, use_fast=False)
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
    # Timestamp per il nuovo modello
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_name = f"{HF_REPO}_{timestamp}"

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
    new_model_dir = "./data/results/final_model"
    print(f"Valutando il nuovo modello da pushare su Hugging Face: {new_model_name}")
    new_f1, new_report = evaluate_model(new_model_dir, X_test_small, y_test_small)
    print("Nuovo modello F1 macro:", new_f1)

    # --- Trova modello precedente se esiste ---
    old_model_path = None
    hub_dir = "./hf_models"
    os.makedirs(hub_dir, exist_ok=True)

    # scarica tutto il repo HF e controlla i nomi delle cartelle
    try:
        snapshot_path = snapshot_download(f"{HF_USERNAME}/{HF_REPO}", local_dir=hub_dir, local_dir_use_symlinks=False)
        # prendi la cartella con prefisso 'trained-model_'
        candidates = [d for d in os.listdir(snapshot_path) if d.startswith(HF_REPO)]
        if candidates:
            candidates.sort(reverse=True)  # ultima versione in cima
            old_model_path = os.path.join(snapshot_path, candidates[0])
            print(f"Trovato modello precedente: {candidates[0]}")
            old_f1, _ = evaluate_model(old_model_path, X_test_small, y_test_small)
            print("Vecchio modello F1 macro:", old_f1)
        else:
            old_f1 = -1
            print("Nessun modello precedente trovato.")
    except Exception:
        old_f1 = -1
        print("Nessun modello precedente trovato o errore nel download.")

    # --- Confronto e push sul Hub ---
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if new_f1 >= old_f1:
        print("Il nuovo modello è migliore o pari al precedente. Lo pushiamo su Hugging Face...")
        model = AutoModelForSequenceClassification.from_pretrained(new_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(new_model_dir)
        trainer.push_to_hub()
        # model.push_to_hub(f"{HF_USERNAME}/{new_model_name}")
        # tokenizer.push_to_hub(f"{HF_USERNAME}/{new_model_name}")
        print(f"Modello pushato su Hugging Face: {new_model_name}")
    else:
        print("Il modello precedente è migliore. Nessun push del nuovo modello.")

    print("\nReport dettagliato del nuovo modello:")
    print(new_report)

if __name__ == "__main__":
    main()
