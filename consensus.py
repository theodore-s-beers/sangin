from collections import Counter
from typing import Any

from infer import PersianMeterClassifier


class MeterAnalyzer:
    def __init__(
        self,
        model_path: str = "./persian-meter-classifier",
        label_map_path: str = "./label_map.json",
    ):
        self.classifier = PersianMeterClassifier(model_path, label_map_path)

    def analyze_meter(
        self, hemistichs: list[str], conf_threshold: float = 0.95
    ) -> dict[str, Any]:
        predictions = self.classifier.predict(hemistichs, top_k=1)

        high_conf_meters: list[str] = []

        for hemistich_preds in predictions:
            meter, confidence = hemistich_preds[0]
            if confidence >= conf_threshold:
                high_conf_meters.append(meter)

        if high_conf_meters:
            counter = Counter(high_conf_meters)
            winner, votes = counter.most_common()[0]

            return {
                "predicted_meter": winner,
                "votes": votes,
                "total_qualifying": len(high_conf_meters),
                "total_hemistichs": len(hemistichs),
                "consensus_ratio": votes / len(high_conf_meters),
                "confidence_threshold": conf_threshold,
                "individual_predictions": [
                    (h, p[0][0], p[0][1]) for h, p in zip(hemistichs, predictions)
                ],
            }
        else:
            return {
                "predicted_meter": None,
                "message": f"No predictions above {conf_threshold:.0%} confidence",
                "confidence_threshold": conf_threshold,
                "individual_predictions": [
                    (h, p[0][0], p[0][1]) for h, p in zip(hemistichs, predictions)
                ],
            }

    def analyze_and_print(
        self, hemistichs: list[str], conf_threshold: float = 0.95
    ) -> dict[str, Any]:
        results = self.analyze_meter(hemistichs, conf_threshold)

        print("=" * 64)
        print("📖 METER ANALYSIS")
        print("=" * 64)

        print("\n📝 Individual predictions:")
        for i, (hemistich, meter, confidence) in enumerate(
            results["individual_predictions"], 1
        ):
            marker = "✓" if confidence >= conf_threshold else "✗"
            print(f"  {i}. {hemistich}")
            print(f"     → {meter} ({confidence:.1%}) {marker}")

        print(f"\n🎯 Consensus (≥{conf_threshold:.0%} confidence):")
        if results["predicted_meter"]:
            print(f"   Predicted meter: {results['predicted_meter']}")
            print(
                f"   Agreement: {results['votes']}/{results['total_qualifying']} qualifying hemistichs ({results['consensus_ratio']:.1%})"
            )
            print(
                f"   Coverage: {results['total_qualifying']}/{results['total_hemistichs']} total hemistichs"
            )
        else:
            print(f"   {results['message']}")
            print("   Consider lowering confidence threshold or checking transcription")

        return results


def main():
    rumi_hemistichs = [
        "بشنو این نی چون شکایت می‌کند",
        "از جدایی‌ها حکایت می‌کند",
        "کز نِیِستان تا مرا بُبریده‌اند",
        "در نفیرم مرد و زن نالیده‌اند",
    ]

    analyzer = MeterAnalyzer()
    analyzer.analyze_and_print(rumi_hemistichs)


if __name__ == "__main__":
    main()
