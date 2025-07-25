import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer import Trainer

df = pd.read_csv("saeb_meters.csv")

label_list = sorted(df["meter_name"].unique())
label_map: dict[str, int] = {label: i for i, label in enumerate(label_list)}
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


# Evaluate each checkpoint
checkpoints = ["checkpoint-30000", "checkpoint-35000", "checkpoint-40000"]

print("Evaluating checkpoints...")
print("=" * 50)

best_accuracy = 0
best_checkpoint = None

for checkpoint in checkpoints:
    print(f"\nEvaluating {checkpoint}...")

    model = AutoModelForSequenceClassification.from_pretrained(
        f"./results/{checkpoint}"
    )

    trainer = Trainer(
        model=model,
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate()

    print(f"Results for {checkpoint}:")
    print(f"  Accuracy: {results['eval_accuracy']:.4f}")
    print(f"  F1: {results['eval_f1']:.4f}")
    print(f"  Loss: {results['eval_loss']:.4f}")

    if results["eval_accuracy"] > best_accuracy:
        best_accuracy = results["eval_accuracy"]
        best_checkpoint = checkpoint

print("\n" + "=" * 50)
print(f"BEST CHECKPOINT: {best_checkpoint}")
print(f"BEST ACCURACY: {best_accuracy:.4f}")
print("\nTo resume training from best checkpoint, use:")
print(f'resume_from_checkpoint="./results/{best_checkpoint}"')
