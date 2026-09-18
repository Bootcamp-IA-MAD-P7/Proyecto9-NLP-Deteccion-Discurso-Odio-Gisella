import streamlit as st
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load trained model and vectorizer
best_rf = joblib.load('models/random_forest_toxic_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
lemmatizer = WordNetLemmatizer()


def tokenize_and_lemmatize(text):
    """
    Tokenize the input text and lemmatize each token.
    """
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(token) for token in tokens]


def predecir_toxicidad(comentario):
    """
    Predict whether a comment is toxic using the trained model.
    Returns the predicted label and the class probabilities.
    """
    tokens = tokenize_and_lemmatize(comentario)
    tokens_text = ' '.join(tokens)
    vector = tfidf.transform([tokens_text])
    prediccion = best_rf.predict(vector)[0]
    probabilidad = best_rf.predict_proba(vector)[0]
    return prediccion, probabilidad


# --- UI ---
st.title("HateShield - Detector de Comentarios Tóxicos")
st.write("Escribe un comentario en inglés para analizar si es tóxico o no.")

comentario = st.text_area("Comentario:", height=100)

if st.button("Analizar"):
    if comentario.strip() == "":
        st.warning("Por favor, escribe un comentario.")
    else:
        prediccion, probabilidad = predecir_toxicidad(comentario)
        if prediccion:
            st.error(f"⚠️ Comentario TÓXICO (confianza: {probabilidad[1]:.0%})")
        else:
            st.success(f"✅ Comentario NO tóxico (confianza: {probabilidad[0]:.0%})")
