import json

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

# Load data (hemistichs already normalized)
# Columns: `hemistich`, `meter_syllables`, `meter_name`, `base_meter`
df = pd.read_csv("hemistichs.csv")

print(f"Dataset size: {len(df)}")
print(f"Number of unique meters: {len(df['meter_name'].unique())}")
print(f"Class distribution:\n{df['meter_name'].value_counts()}")
print("\n" + "=" * 64 + "\n")

# Map meter labels to ints
label_list = sorted(df["meter_name"].unique())
label_map: dict[str, int] = {label: i for i, label in enumerate(label_list)}
print(f"Labels: {list(label_map.keys())[:5]}...")  # Show first 5
df["label"] = df["meter_name"].map(label_map)

# Save label mapping
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)
print("Label mapping saved to label_map.json")

# HuggingFace Dataset
dataset = Dataset.from_pandas(df[["hemistich", "label"]])
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Tokenizer & model
model_name = "FacebookAI/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
assert tokenizer is not None

id2label = {i: label for label, i in label_map.items()}
label2id = label_map

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
)


# Tokenize
def tokenize(examples):
    assert tokenizer is not None
    return tokenizer(examples["hemistich"], truncation=True, padding="max_length")


tokenized = dataset.map(tokenize, batched=True)

use_cuda = torch.cuda.is_available()
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
use_fp16 = use_cuda and not use_bf16

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    save_total_limit=2,
    bf16=use_bf16,
    fp16=use_fp16,
    report_to="none",
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=labels, y_pred=predictions, average="weighted", zero_division=0
    )
    acc = accuracy_score(y_true=labels, y_pred=predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

# Save model and tokenizer
trainer.save_model("./persian-meter-classifier")
tokenizer.save_pretrained("./persian-meter-classifier")
print("Model saved to ./persian-meter-classifier")
