import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PersianMeterClassifier:
    def __init__(
        self,
        model_path: str = "./persian-meter-classifier",
        label_map_path: str = "./label_map.json",
    ):
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)

        # Create reverse mapping (id -> label)
        self.id_to_label = {v: k for k, v in self.label_map.items()}

        print(f"Model loaded! Can classify {len(self.label_map)} different meters:")

        for meter, _ in list(self.label_map.items())[:5]:
            print(f"  - {meter}")
        if len(self.label_map) > 5:
            print(f"  ... and {len(self.label_map) - 5} more")
        print()

    def predict(
        self, hemistichs: list[str], top_k: int = 3
    ) -> list[list[tuple[str, float]]]:
        inputs = self.tokenizer(
            hemistichs, truncation=True, padding=True, return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        results = []
        for i, _ in enumerate(hemistichs):
            top_probs, top_indices = torch.topk(probabilities[i], top_k)

            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                meter_name = self.id_to_label[idx.item()]
                confidence = prob.item()
                predictions.append((meter_name, confidence))

            results.append(predictions)

        return results

    def predict_single(self, hemistich: str, top_k: int = 3) -> list[tuple[str, float]]:
        return self.predict([hemistich], top_k)[0]


def main():
    classifier = PersianMeterClassifier()

    test_hemistichs = [
        "اگر آن تُرکِ شیرازی به‌‌ دست‌ آرَد دلِ ما را",
        "به خال هِندویَش بَخشَم سَمَرقند و بُخارا را",
        "بده ساقی مِیِ باقی که در جَنَّت نخواهی یافت",
        "کنارِ آبِ رُکناباد و گُل‌گَشتِ مُصَلّا را",
        "اگر تندبادی براید ز کنج",
        "بخاک افگند نارسیده ترنج",
        "ستمکاره خوانیمش ار دادگر",
        "هنرمند دانیمش ار بی‌هنر",
    ]

    print("🔍 Classifying hemistichs...\n")

    for hemistich in test_hemistichs:
        print(f"📝 Hemistich: {hemistich}")
        predictions = classifier.predict_single(hemistich, top_k=3)

        print("📊 Predictions:")
        for i, (meter, confidence) in enumerate(predictions, 1):
            print(f"  {i}. {meter:<30} ({confidence:.1%})")
        print("-" * 64)


if __name__ == "__main__":
    main()
