"""Exploration du dataset movie_reviews : taille, longueur des critiques, mots fréquents."""
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import movie_reviews

from preprocessing import clean_tokens

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"


def main():
    for category in movie_reviews.categories():
        fileids = movie_reviews.fileids(category)
        avg_len = sum(len(movie_reviews.words(f)) for f in fileids) / len(fileids)
        print(f"{category} : {len(fileids)} critiques, {avg_len:.0f} mots en moyenne")

    freq = {}
    for category in movie_reviews.categories():
        words = clean_tokens(movie_reviews.words(categories=category))
        freq[category] = nltk.FreqDist(words)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    titles = {"pos": "Critiques positives", "neg": "Critiques négatives"}
    for ax, category in zip(axes, ["pos", "neg"]):
        top = freq[category].most_common(15)
        ax.barh([w for w, _ in top][::-1], [c for _, c in top][::-1])
        ax.set_title(titles[category])
        ax.set_xlabel("Nombre d'occurrences")
    fig.suptitle("Mots les plus fréquents après nettoyage")
    fig.tight_layout()
    IMAGES_DIR.mkdir(exist_ok=True)
    fig.savefig(IMAGES_DIR / "top_words.png", dpi=120, bbox_inches="tight")
    print("Figure enregistrée : images/top_words.png")
    plt.show()


if __name__ == "__main__":
    main()
