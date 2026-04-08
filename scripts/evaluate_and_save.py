# scripts/evaluate_and_save.py
import torch
import os
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report
from huggingface_hub import snapshot_download

HF_USERNAME  = "andrealuigipala"
HF_REPO      = "trained-model"
MODEL_DIR    = "./data/results/trained_model"
DATA_URL     = "https://raw.githubusercontent.com/andrealuigipala-del/MLOps_sentiment_project/refs/heads/main/data/src/Twitter_Data.csv"
SAMPLE_SIZE  = 150


def evaluate_model(model_dir, texts, labels, batch_size=16):
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

    f1     = f1_score(labels, all_preds, average="macro")
    report = classification_report(labels, all_preds, digits=4)
    return round(f1, 4), report


def main():
    # 1. Carica dataset — shift +1 coerente con preprocess.py
    df = pd.read_csv(DATA_URL)
    df = df.dropna(subset=['clean_text', 'category'])
    df['clean_text'] = df['clean_text'].astype(str)
    df['category']   = df['category'] + 1      # FIX: era mancante
    df['category']   = df['category'].astype(int)

    sample  = df.sample(SAMPLE_SIZE, random_state=99)
    texts   = sample['clean_text'].tolist()
    labels  = sample['category'].tolist()

    # 2. Valuta nuovo modello locale
    print(f"Valutando modello locale: {MODEL_DIR}")
    new_f1, new_report = evaluate_model(MODEL_DIR, texts, labels)
    print(f"Nuovo modello — F1 macro: {new_f1}")

    # 3. Scarica e valuta modello precedente da HuggingFace
    old_f1  = -1
    hub_dir = "./hf_models"
    os.makedirs(hub_dir, exist_ok=True)

    try:
        snapshot_path = snapshot_download(
            f"{HF_USERNAME}/{HF_REPO}",
            local_dir=hub_dir,
            local_dir_use_symlinks=False
        )
        print(f"Modello precedente trovato su HF: {HF_REPO}")
        old_f1, _ = evaluate_model(snapshot_path, texts, labels)
        print(f"Modello precedente — F1 macro: {old_f1}")
    except Exception:
        print("Nessun modello precedente su HuggingFace.")

    # 4. Confronto e push condizionale
    if new_f1 >= old_f1:
        print("Nuovo modello migliore o pari. Push su HuggingFace...")
        hf_token = os.environ.get("HF_TOKEN")
        model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model.push_to_hub(f"{HF_USERNAME}/{HF_REPO}", token=hf_token)
        tokenizer.push_to_hub(f"{HF_USERNAME}/{HF_REPO}", token=hf_token)
        print("Push completato.")
    else:
        print("Modello precedente migliore. Nessun push.")

    print("\nReport nuovo modello:")
    print(new_report)


if __name__ == "__main__":
    main()
