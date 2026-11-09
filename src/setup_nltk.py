"""Télécharge les ressources NLTK nécessaires au projet (à lancer une seule fois)."""
import nltk

RESOURCES = [
    "movie_reviews",  # 2 000 critiques de films étiquetées positives / négatives
    "stopwords",      # mots vides : "the", "a", "is"...
    "wordnet",        # dictionnaire utilisé par le lemmatiseur
    "omw-1.4",
    "punkt",          # découpage d'une phrase en mots
    "punkt_tab",
]

if __name__ == "__main__":
    for name in RESOURCES:
        print(f"Téléchargement de {name}...")
        nltk.download(name, quiet=True)
    print("Ressources NLTK prêtes.")
