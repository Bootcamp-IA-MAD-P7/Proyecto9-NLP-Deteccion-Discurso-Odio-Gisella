"""Unit tests for app/utils.py: text cleaning, tokenization, and toxicity prediction."""
from utils import clean_text, predecir_toxicidad, tokenize_and_lemmatize


class FakeEstimator:
    """Minimal stand-in for one fitted sklearn estimator inside the ensemble."""

    def __init__(self, prediction):
        self._prediction = prediction

    def predict(self, X):
        return [self._prediction]


class FakeEnsemble:
    """Minimal stand-in for the VotingClassifier ensemble (hard voting)."""

    def __init__(self, base_predictions, final_prediction):
        self.named_estimators_ = {
            f"model_{i}": FakeEstimator(pred)
            for i, pred in enumerate(base_predictions)
        }
        self._final_prediction = final_prediction

    def predict(self, X):
        return [self._final_prediction]


class FakeTfidf:
    """Minimal stand-in for the fitted TfidfVectorizer; passes text through untouched."""

    def transform(self, texts):
        return texts


# --- clean_text ---

def test_clean_text_lowercases_and_removes_punctuation():
    result = clean_text("This IS Very Bad!!!")
    assert result == result.lower()
    assert "!" not in result


def test_clean_text_removes_urls():
    result = clean_text("check this out http://example.com/video now")
    assert "http" not in result
    assert "example" not in result


def test_clean_text_removes_mentions():
    result = clean_text("@someuser this is unacceptable")
    assert "someuser" not in result


def test_clean_text_keeps_negations_and_intensifiers():
    # These carry meaning for hate speech detection and must never be stripped
    result = clean_text("this is not very good")
    assert "not" in result
    assert "very" in result


def test_clean_text_removes_common_stopwords():
    result = clean_text("this is the best video of all time")
    assert "the" not in result
    assert "of" not in result


def test_clean_text_empty_string_returns_empty_string():
    assert clean_text("") == ""


def test_clean_text_only_stopwords_returns_empty_string():
    # every token here is a stopword that is NOT a negation/intensifier
    assert clean_text("the a an of") == ""


# --- tokenize_and_lemmatize ---

def test_tokenize_and_lemmatize_splits_into_tokens():
    tokens = tokenize_and_lemmatize("bad videos everywhere")
    assert isinstance(tokens, list)
    assert len(tokens) == 3


def test_tokenize_and_lemmatize_reduces_plurals_to_singular():
    tokens = tokenize_and_lemmatize("these videos are the worst")
    assert "video" in tokens


def test_tokenize_and_lemmatize_empty_string_returns_empty_list():
    assert tokenize_and_lemmatize("") == []


# --- predecir_toxicidad ---

def test_predecir_toxicidad_returns_expected_keys():
    ensemble = FakeEnsemble(base_predictions=[1, 1, 0], final_prediction=1)
    tfidf = FakeTfidf()

    result = predecir_toxicidad("you are terrible", ensemble, tfidf)

    assert set(result.keys()) == {"es_toxico", "votos_toxico", "total_modelos"}


def test_predecir_toxicidad_toxic_case():
    ensemble = FakeEnsemble(base_predictions=[1, 1, 1], final_prediction=1)
    tfidf = FakeTfidf()

    result = predecir_toxicidad("you are terrible", ensemble, tfidf)

    assert result["es_toxico"] is True
    assert result["votos_toxico"] == 3
    assert result["total_modelos"] == 3


def test_predecir_toxicidad_non_toxic_case():
    ensemble = FakeEnsemble(base_predictions=[0, 0, 0], final_prediction=0)
    tfidf = FakeTfidf()

    result = predecir_toxicidad("thank you for the video", ensemble, tfidf)

    assert result["es_toxico"] is False
    assert result["votos_toxico"] == 0


def test_predecir_toxicidad_counts_votes_independently_of_final_prediction():
    # Hard voting: the final prediction is the majority vote, but votos_toxico
    # must still reflect exactly how many base models voted toxic.
    ensemble = FakeEnsemble(base_predictions=[1, 0, 0], final_prediction=0)
    tfidf = FakeTfidf()

    result = predecir_toxicidad("some comment", ensemble, tfidf)

    assert result["votos_toxico"] == 1
    assert result["es_toxico"] is False
