"""
Script per caricare un modello di analisi del sentiment già addestrato da HuggingFace Hub
e usarlo per predizioni su nuovi testi.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Nome del modello sul HuggingFace Hub
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def load_model():
    """Carica il modello e il tokenizer dal Hub"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_analyzer

def predict_sentiment(texts):
    """Predice il sentiment di una lista di testi"""
    sentiment_analyzer = load_model()
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
