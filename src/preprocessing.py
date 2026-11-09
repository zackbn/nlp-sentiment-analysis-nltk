"""Prétraitement du texte : nettoyage, mots vides, lemmatisation."""
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# On garde les négations : "not good" et "good" n'ont pas le même sens.
NEGATIONS = {"not", "no", "nor"}
STOP_WORDS = set(stopwords.words("english")) - NEGATIONS
LEMMATIZER = WordNetLemmatizer()


def clean_tokens(tokens):
    """Met en minuscules, garde les mots alphabétiques, retire les mots vides et lemmatise.

    Exemple : ["The", "movies", "were", "NOT", "great", "!"] -> ["movie", "not", "great"]
    """
    cleaned = []
    for token in tokens:
        word = token.lower()
        if not word.isalpha() or word in STOP_WORDS:
            continue
        cleaned.append(LEMMATIZER.lemmatize(word))
    return cleaned


def preprocess_text(text):
    """Découpe une phrase en mots (tokenisation) puis la nettoie."""
    return clean_tokens(word_tokenize(text))


def document_features(words, vocabulary):
    """Sac de mots : pour chaque mot du vocabulaire, indique s'il est présent dans le document."""
    words = set(words)
    return {f"contient({w})": (w in words) for w in vocabulary}
