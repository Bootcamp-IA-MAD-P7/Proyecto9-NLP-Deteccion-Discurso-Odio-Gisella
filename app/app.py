import streamlit as st
import joblib
from utils import predecir_toxicidad

# Load trained model and vectorizer
ensemble = joblib.load('models/ensemble_toxic_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# --- UI ---
st.title("HateShield - Detector de Comentarios Tóxicos")
st.write("Escribe un comentario en inglés para analizar si es tóxico o no.")

comentario = st.text_area("Comentario:", height=100)

if st.button("Analizar"):
    if comentario.strip() == "":
        st.warning("Por favor, escribe un comentario.")
    else:
        resultado = predecir_toxicidad(comentario, ensemble, tfidf)
        votos = resultado["votos_toxico"]
        total = resultado["total_modelos"]
        if resultado["es_toxico"]:
            st.error(f"⚠️ Comentario TÓXICO ({votos}/{total} modelos coinciden)")
        else:
            st.success(f"✅ Comentario NO tóxico ({total - votos}/{total} modelos coinciden)")