# scripts/train.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import os

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def fine_tune_model(X_train, y_train, X_test, y_test,
                    output_dir="./results",
                    epochs=1,
                    batch_size=16):
    """
    Fine-tuning del modello con freeze del backbone RoBERTa.
    Solo il classification head viene aggiornato — molto più veloce.
    """

    # 1. Tokenizer e modello
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    # 2. Freeze del backbone — alleniamo solo il classifier head
    for param in model.roberta.parameters():
        param.requires_grad = False

    # 3. Tokenizzazione con max_length=64 (tweet sono corti)
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=64
        )

    train_dataset = Dataset.from_dict({
        "text": X_train.tolist(),
        "labels": y_train.tolist()
    })
    test_dataset = Dataset.from_dict({
        "text": X_test.tolist(),
        "labels": y_test.tolist()
    })

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    test_dataset  = test_dataset.map(tokenize_fn, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch",  columns=["input_ids", "attention_mask", "labels"])

    # 4. TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        push_to_hub=False,
    )

    # 5. Compute metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": float(accuracy)}

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # 7. Salvataggio
    save_path = "./data/results/trained_model"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Modello salvato in {save_path}")

    return trainer, model, tokenizer


if __name__ == "__main__":
    from scripts.preprocess import load_dataset

    DATA_URL = "https://raw.githubusercontent.com/andrealuigipala-del/MLOps_sentiment_project/refs/heads/main/data/src/Twitter_Data.csv"

    X_train, X_test, y_train, y_test = load_dataset(DATA_URL)

    # 2000 campioni: statisticamente significativi, veloci con freeze
    fine_tune_model(
        X_train[:2000], y_train[:2000],
        X_test[:400],   y_test[:400]
    )
