import json

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer import Trainer

with open("label_map.json", "r") as f:
    label_map = json.load(f)

# Reverse label mapping
id_to_label = {v: k for k, v in label_map.items()}

df = pd.read_csv("hemistichs.csv")
df["label"] = df["meter_name"].map(label_map)

subset_df = df[["hemistich", "label"]]
dataset = Dataset.from_pandas(subset_df)
dataset = dataset.train_test_split(test_size=0.1)

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")


def tokenize(examples):
    return tokenizer(examples["hemistich"], truncation=True, padding="max_length")


tokenized = dataset.map(tokenize, batched=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=labels, y_pred=predictions, average="weighted", zero_division=0
    )
    acc = accuracy_score(y_true=labels, y_pred=predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


model_path = "./persian-meter-classifier-v2"
print(f"Loading model from {model_path}...")

model = AutoModelForSequenceClassification.from_pretrained(model_path)

trainer = Trainer(
    model=model, eval_dataset=tokenized["test"], compute_metrics=compute_metrics
)

print("Running evaluation...")
print("=" * 64)

results = trainer.evaluate()

print("Final model results:")
print(f"  Accuracy: {results['eval_accuracy']:.4f}")
print(f"  F1: {results['eval_f1']:.4f}")
print(f"  Precision: {results['eval_precision']:.4f}")
print(f"  Recall: {results['eval_recall']:.4f}")
print(f"  Loss: {results['eval_loss']:.4f}")

print("\n" + "=" * 64)
print("Evaluation complete!")
