from collections import Counter

from infer import PersianMeterClassifier


class PoemMeterAnalyzer:
    def __init__(
        self,
        model_path: str = "./persian-meter-classifier",
        label_map_path: str = "./label_map.json",
    ):
        self.classifier = PersianMeterClassifier(model_path, label_map_path)

    def analyze_poem(self, hemistichs: list[str]) -> dict:
        all_predictions = self.classifier.predict(hemistichs, top_k=3)

        results = {
            "hemistichs": hemistichs,
            "individual_predictions": all_predictions,
            "analysis": {},
        }

        results["analysis"]["consensus"] = self._consensus_analysis(all_predictions)

        return results

    def _consensus_analysis(
        self, all_predictions: list[list[tuple[str, float]]]
    ) -> dict:
        high_confidence_predictions = []
        confidence_threshold = 0.95

        for hemistich_preds in all_predictions:
            top_meter, top_conf = hemistich_preds[0]
            if top_conf >= confidence_threshold:
                high_confidence_predictions.append(top_meter)

        if high_confidence_predictions:
            counter = Counter(high_confidence_predictions)
            most_common = counter.most_common()[0]

            return {
                "predicted_meter": most_common[0],
                "high_confidence_votes": most_common[1],
                "high_confidence_total": len(high_confidence_predictions),
                "consensus_ratio": most_common[1] / len(high_confidence_predictions)
                if high_confidence_predictions
                else 0,
                "confidence_threshold": confidence_threshold,
            }
        else:
            return {
                "predicted_meter": None,
                "message": f"No predictions above {confidence_threshold:.0%} confidence",
                "confidence_threshold": confidence_threshold,
            }

    def print_analysis(self, results: dict):
        print("=" * 64)
        print("📖 POEM-LEVEL METER ANALYSIS")
        print("=" * 64)

        print(f"\n📝 Analyzing {len(results['hemistichs'])} hemistichs:")
        for i, hemistich in enumerate(results["hemistichs"], 1):
            top_pred = results["individual_predictions"][i - 1][0]
            print(f"  {i}. {hemistich}")
            print(f"     → {top_pred[0]} ({top_pred[1]:.1%})")

        print("\n" + "=" * 64)
        print("📊 ANALYSIS RESULTS")
        print("=" * 64)

        cons = results["analysis"]["consensus"]
        print("\n🎯 HIGH-CONFIDENCE CONSENSUS:")
        if cons["predicted_meter"]:
            print(f"   Predicted meter: {cons['predicted_meter']}")
            print(
                f"   Consensus: {cons['high_confidence_votes']}/{cons['high_confidence_total']} high-conf predictions ({cons['consensus_ratio']:.1%})"
            )
        else:
            print(f"   {cons['message']}")


def main():
    analyzer = PoemMeterAnalyzer()

    rumi_hemistichs = [
        "بشنو این نی چون شکایت می‌کند",
        "از جدایی‌ها حکایت می‌کند",
        "کز نِیِستان تا مرا بُبریده‌اند",
        "در نفیرم مرد و زن نالیده‌اند",
    ]

    results = analyzer.analyze_poem(rumi_hemistichs)
    analyzer.print_analysis(results)


if __name__ == "__main__":
    main()
