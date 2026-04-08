# scripts/deploy_huggingface.py
"""
Carica il modello fine-tunato da HuggingFace Hub
e lo usa per predizioni su nuovi testi.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Il tuo modello fine-tunato — NON il modello base Cardiff
HF_MODEL_ID = "andrealuigipala/trained-model"

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model     = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    analyzer  = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return analyzer


def predict_sentiment(texts):
    analyzer = load_model()
    raw      = analyzer(texts)
    # Traduce LABEL_0/1/2 in etichette leggibili
    return [
        {"text": t, "sentiment": LABEL_MAP.get(r["label"], r["label"]), "score": round(r["score"], 4)}
        for t, r in zip(texts, raw)
    ]


if __name__ == "__main__":
    samples = [
        "MachineInnovators Inc. is absolutely amazing!",
        "I am not happy with their customer support.",
        "The service is okay, nothing special."
    ]
    results = predict_sentiment(samples)
    for r in results:
        print(f"[{r['sentiment'].upper()}] ({r['score']}) — {r['text']}")
