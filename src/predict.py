"""Prédit le sentiment de nouvelles phrases avec le modèle entraîné.

Exemples :
    python src/predict.py
    python src/predict.py "This movie was a masterpiece" "Boring and too long"
"""
import argparse
import pickle

from preprocessing import document_features, preprocess_text
from train import MODEL_PATH

EXAMPLES = [
    "An amazing film with brilliant acting and a beautiful story.",
    "The plot was boring and the characters were terrible.",
    "I did not like this movie at all, a complete waste of time.",
]


def main(sentences):
    if not MODEL_PATH.exists():
        raise SystemExit("Modèle introuvable : lancez d'abord  python src/train.py")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    classifier, vocabulary = model["classifier"], model["vocabulary"]

    for sentence in sentences:
        features = document_features(preprocess_text(sentence), vocabulary)
        proba_pos = classifier.prob_classify(features).prob("pos")
        label = "POSITIF" if proba_pos >= 0.5 else "NÉGATIF"
        print(f"{label:<8} ({proba_pos:.0%} positif) | {sentence}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse de sentiments d'une phrase en anglais")
    parser.add_argument("sentences", nargs="*", help="phrases à analyser (en anglais)")
    args = parser.parse_args()
    main(args.sentences or EXAMPLES)
