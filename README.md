# Analyse de sentiments avec NLTK

Classification de critiques de films en **positives** ou **négatives** avec Python et NLTK (Natural Language Toolkit).

```
$ python src/predict.py "An amazing film with brilliant acting" "Boring and too long"
POSITIF  (..% positif) | An amazing film with brilliant acting
NÉGATIF  (..% positif) | Boring and too long
```
<!-- Remplacez l'exemple ci-dessus par la vraie sortie de predict.py -->

## Dataset

**movie_reviews** (corpus NLTK) : 2 000 critiques de films en anglais, 1 000 positives et 1 000 négatives.

## Pipeline NLP

1. **Tokenisation** : découpage du texte en mots (`word_tokenize`).
2. **Nettoyage** : minuscules, suppression de la ponctuation et des nombres.
3. **Mots vides** : suppression des mots sans valeur de sentiment (« the », « is »…), en **gardant les négations** (« not », « no ») qui inversent le sens.
4. **Lemmatisation** : réduction de chaque mot à sa forme de base (« movies » → « movie ») avec WordNet.
5. **Sac de mots** : chaque critique devient un vecteur indiquant la présence des 2 000 mots les plus fréquents du jeu d'entraînement.
6. **Classification** : Naive Bayes (`nltk.NaiveBayesClassifier`), entraîné sur 80 % des critiques et évalué sur les 20 % restantes.

## Résultats

<!-- Après  python src/train.py : collez ici le contenu de results/metrics.txt -->

| Matrice de confusion | Mots les plus fréquents |
| --- | --- |
| ![Matrice de confusion](images/confusion_matrix.png) | ![Mots fréquents](images/top_words.png) |

<!-- Collez ici les 10 premières lignes de « Mots les plus discriminants » affichées par train.py -->

## Installation

```bash
git clone https://github.com/zackbn/nlp-sentiment-analysis-nltk.git
cd nlp-sentiment-analysis-nltk
pip install -r requirements.txt
python src/setup_nltk.py      # télécharge les ressources NLTK (une seule fois)
```

## Utilisation

```bash
python src/explore.py         # statistiques et mots fréquents
python src/train.py           # entraînement, évaluation, sauvegarde du modèle
python src/predict.py "Your sentence here"
```

## Structure

```
nlp-sentiment-analysis-nltk/
├── src/
│   ├── setup_nltk.py       # téléchargement des ressources NLTK
│   ├── preprocessing.py    # tokenisation, nettoyage, lemmatisation, sac de mots
│   ├── explore.py          # exploration du dataset
│   ├── train.py            # entraînement et évaluation
│   └── predict.py          # prédiction sur de nouvelles phrases
├── images/                 # figures générées
├── results/                # métriques
├── models/                 # modèle entraîné (non versionné)
└── requirements.txt
```

## Pistes d'amélioration

- Représentation TF-IDF et comparaison avec la régression logistique (scikit-learn).
- Prise en compte des bigrammes (« not good »).
- Modèle Transformer pré-entraîné (BERT) avec PyTorch.

## Auteur

**Zakarya Benani Sadoun** — Étudiant en Master IA & Big Data (ESGI Paris), à la recherche d'une alternance Ingénieur IA · [GitHub](https://github.com/zackbn)
