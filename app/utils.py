"""Shared preprocessing and prediction utilities for HateShield."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Keep negations and intensifiers, since they carry meaning
# that is critical for hate speech detection
NEGATIONS_AND_INTENSIFIERS = {
    'no', 'not', 'nor', 'never', 'none', 'nothing', 'nowhere',
    'neither', 'very', 'too', 'so', 'only', 'just'
}
STOP_WORDS = set(stopwords.words('english')) - NEGATIONS_AND_INTENSIFIERS
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean raw text for NLP processing:
    - Lowercase
    - Remove URLs, mentions, and special characters
    - Remove stopwords (excluding negations/intensifiers)
    - Lemmatize tokens
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


def tokenize_and_lemmatize(text: str) -> list[str]:
    """Tokenize cleaned text and lemmatize each token."""
    tokens = word_tokenize(text)
    return [LEMMATIZER.lemmatize(token) for token in tokens]


def predecir_toxicidad(comentario: str, ensemble, tfidf) -> dict:
    """
    Predict whether a comment is toxic using the trained ensemble.
    Returns the final prediction plus how many base models voted toxic
    (a hard-voting ensemble has no predict_proba, so this is used
    instead as an interpretable confidence signal).
    """
    texto_limpio = clean_text(comentario)
    tokens = tokenize_and_lemmatize(texto_limpio)
    tokens_text = ' '.join(tokens)
    vector = tfidf.transform([tokens_text])

    prediccion_final = ensemble.predict(vector)[0]

    votos = [
        int(estimator.predict(vector)[0])
        for _, estimator in ensemble.named_estimators_.items()
    ]
    votos_toxico = sum(votos)

    return {
        "es_toxico": bool(prediccion_final),
        "votos_toxico": votos_toxico,
        "total_modelos": len(votos),
    }