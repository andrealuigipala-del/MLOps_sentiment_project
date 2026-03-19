# scripts/train.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np

def fine_tune_model(X_train, y_train, X_test, y_test, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    output_dir="./results", epochs=3, batch_size=16):
    """
    Fine-tuning di un modello HuggingFace per l'analisi del sentiment.

    Args:
        X_train, y_train: dati di training (pd.Series)
        X_test, y_test: dati di test/valutazione (pd.Series)
        model_name (str): nome modello su HuggingFace Hub
        output_dir (str): cartella per salvare i pesi del modello
        epochs (int): numero di epoche di training
        batch_size (int): batch size per training ed eval

    Returns:
        trainer: oggetto Trainer già allenato
        model: modello fine-tunato
        tokenizer: tokenizer del modello
    """

    # 1. Tokenizer e modello
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,  # negative, neutral, positive
    )

    # 2. Prepara dataset HuggingFace
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = Dataset.from_dict({"text": X_train.tolist(), "labels": y_train.tolist()})
    test_dataset = Dataset.from_dict({"text": X_test.tolist(), "labels": y_test.tolist()})

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    # HuggingFace richiede che i tensori labels siano int
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 3. TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        push_to_hub=False  # eventualmente True se vuoi caricare su HuggingFace
    )

    # 4. Trainer
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 5. Avvia fine-tuning
    trainer.train()

    return trainer, model, tokenizer

# Esempio di utilizzo
if __name__ == "__main__":
    from scripts.preprocess import load_dataset

    url = "https://raw.githubusercontent.com/andrealuigipala-del/MLOps_Final_Project/refs/heads/main/Twitter_Data.csv"
    X_train, X_test, y_train, y_test = load_dataset(url)

    trainer, model, tokenizer = fine_tune_model(X_train, y_train, X_test, y_test)
