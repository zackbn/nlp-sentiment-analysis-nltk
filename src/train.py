"""Entraîne un classifieur Naive Bayes (NLTK) pour l'analyse de sentiments.

Étapes : chargement -> nettoyage -> découpage train/test -> vocabulaire
-> sac de mots -> entraînement -> évaluation -> sauvegarde du modèle.
"""
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import movie_reviews
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from preprocessing import clean_tokens, document_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sentiment_nb.pkl"
VOCAB_SIZE = 2000
TEST_RATIO = 0.2
RANDOM_SEED = 42


def load_documents():
    """Renvoie une liste de (mots nettoyés, étiquette) pour les 2 000 critiques."""
    documents = [
        (clean_tokens(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]
    random.Random(RANDOM_SEED).shuffle(documents)
    return documents


def main():
    documents = load_documents()
    n_test = int(len(documents) * TEST_RATIO)
    train_docs, test_docs = documents[n_test:], documents[:n_test]
    print(f"Train : {len(train_docs)} critiques | Test : {len(test_docs)} critiques")

    # Le vocabulaire est construit sur le train uniquement (pas de fuite de données)
    all_words = nltk.FreqDist(w for words, _ in train_docs for w in words)
    vocabulary = [w for w, _ in all_words.most_common(VOCAB_SIZE)]

    train_set = [(document_features(words, vocabulary), label) for words, label in train_docs]
    test_set = [(document_features(words, vocabulary), label) for words, label in test_docs]

    classifier = nltk.NaiveBayesClassifier.train(train_set)

    # Évaluation
    accuracy = nltk.classify.accuracy(classifier, test_set)
    y_true = [label for _, label in test_set]
    y_pred = [classifier.classify(features) for features, _ in test_set]
    print(f"\nAccuracy sur le test : {accuracy:.3f}\n")
    report = classification_report(y_true, y_pred, target_names=["neg", "pos"])
    print(report)

    print("Mots les plus discriminants :")
    classifier.show_most_informative_features(15)

    # Résultats sauvegardés
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics.txt").write_text(
        f"Accuracy : {accuracy:.3f}\n\n{report}", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=["neg", "pos"], display_labels=["Négatif", "Positif"],
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"Naive Bayes — accuracy = {accuracy:.3f}")
    (ROOT / "images").mkdir(exist_ok=True)
    fig.savefig(ROOT / "images" / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    print("\nFigure enregistrée : images/confusion_matrix.png")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"classifier": classifier, "vocabulary": vocabulary}, f)
    print(f"Modèle enregistré : {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
