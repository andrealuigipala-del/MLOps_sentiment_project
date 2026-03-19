from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Carica tokenizer e modello dal HuggingFace Hub
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
