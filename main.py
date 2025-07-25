import json

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

# Load data (hemistichs already normalized)
# Columns: `hemistich`, `meter_syllables`, `meter_name`, `base_meter`
df = pd.read_csv("saeb_meters.csv")

print(f"Dataset size: {len(df)}")
print(f"Number of unique meters: {len(df['meter_name'].unique())}")
print(f"Class distribution:\n{df['meter_name'].value_counts()}")
print("\n" + "=" * 40 + "\n")

# Map meter labels to ints
label_list = sorted(df["meter_name"].unique())
label_map: dict[str, int] = {label: i for i, label in enumerate(label_list)}
print(f"Labels: {list(label_map.keys())[:5]}...")  # Show first 5
df["label"] = df["meter_name"].replace(label_map)

# Save label mapping
with open("label_map.json", "w") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)
print("Label mapping saved to label_map.json")

# HuggingFace Dataset
subset_df = df[["hemistich", "label"]]
assert isinstance(subset_df, pd.DataFrame)  # To placate Pyright
dataset = Dataset.from_pandas(subset_df)
dataset = dataset.train_test_split(test_size=0.1)

# Tokenizer & model
model_name = "FacebookAI/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label_list)
)


# Tokenize
def tokenize(examples):
    return tokenizer(examples["hemistich"], truncation=True, padding="max_length")


tokenized = dataset.map(tokenize, batched=True)

# Train
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=labels, y_pred=predictions, average="weighted"
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
