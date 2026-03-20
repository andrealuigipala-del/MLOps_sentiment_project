# scripts/evaluate_and_save.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import shutil
from datetime import datetime

# Configurazione fissa
MODEL_DIR = "./final_model"
NEW_MODEL_DIR = "./temp_model"  # dove dovrebbe essere il modello appena allenato
DATA_PATH = "https://raw.githubusercontent.com/andrealuigipala-del/MLOps_Final_Project/refs/heads/main/Twitter_Data.csv"

def evaluate_model(model_dir, X, y):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    inputs = tokenizer(X, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).tolist()
    
    f1_macro = f1_score(y, preds, average="macro")
    report = classification_report(y, preds, digits=4)
    return f1_macro, report

def main():
    # Controlla se esiste un nuovo modello
    if not os.path.exists(NEW_MODEL_DIR):
        print(f"ATTENZIONE: Nessun modello nuovo trovato in {NEW_MODEL_DIR}. Esco.")
        return

    # Carica dataset di test
    df = pd.read_csv(DATA_PATH)
    X_test = df['text'].tolist()
    y_test = df['category'].tolist()

    # Valuta il nuovo modello
    print("Valutando il modello appena addestrato...")
    new_f1, new_report = evaluate_model(NEW_MODEL_DIR, X_test, y_test)
    print("Nuovo modello F1 macro:", new_f1)

    # Controlla se esiste il modello vecchio
    if os.path.exists(MODEL_DIR):
        print("Valutando il modello salvato precedentemente...")
        old_f1, _ = evaluate_model(MODEL_DIR, X_test, y_test)
        print("Vecchio modello F1 macro:", old_f1)
    else:
        old_f1 = -1  # non esiste, quindi il nuovo modello va salvato

    # Confronto e salvataggio
    if new_f1 > old_f1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.path.exists(MODEL_DIR):
            backup_dir = f"./backup_model_{timestamp}"
            shutil.move(MODEL_DIR, backup_dir)
            print(f"Vecchio modello salvato in {backup_dir}")
        shutil.move(NEW_MODEL_DIR, MODEL_DIR)
        print("Nuovo modello salvato in final_model/")
    else:
        print("Il nuovo modello NON supera quello precedente. Nessuna sovrascrittura.")

    print("\nReport dettagliato del nuovo modello:")
    print(new_report)

if __name__ == "__main__":
    main()
