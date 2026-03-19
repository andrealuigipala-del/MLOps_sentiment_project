"""
Script per caricare un modello locale di analisi del sentiment già allenato
e usarlo per predizioni su nuovi testi.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Percorso locale dove il modello è stato salvato
MODEL_PATH = "./models/sentiment_model"

def load_local_model():
    """Carica il modello salvato localmente"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_analyzer

def predict_sentiment(texts):
    """Predice il sentiment di una lista di testi"""
    sentiment_analyzer = load_local_model()
    return sentiment_analyzer(texts)

if __name__ == "__main__":
    # Esempio di utilizzo
    sample_texts = [
        "MachineInnovators Inc. is amazing!",
        "I am not happy with their support.",
        "The service is okay, nothing special."
    ]
    predictions = predict_sentiment(sample_texts)
    print(predictions)
