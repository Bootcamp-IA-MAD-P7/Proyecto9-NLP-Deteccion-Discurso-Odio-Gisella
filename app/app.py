import pandas as pd
import streamlit as st
import joblib
from utils import predecir_toxicidad
from youtube_utils import analyze_youtube_video

# Load trained model and vectorizer
ensemble = joblib.load('../models/ensemble_toxic_model.pkl')
tfidf = joblib.load('../models/tfidf_vectorizer.pkl')


# --- UI ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #f5f0e6 0%, #ede4d3 100%);
    }
    h1 {
        color: #0f5c5c;
    }
    p, label {
        color: #3d3427;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #c9b99a;
        background-color: #fffdf8;
    }
    div.stButton > button {
        background-color: #17a2a2;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.5em;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #0f7a7a;
        color: white;
    }
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_icon="🛡️", page_title="HateShield")

st.title("🛡️ HateShield - Detector de Comentarios Tóxicos")
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


# --- Análisis de comentarios de YouTube ---
st.divider()
st.subheader("📺 Analizar comentarios de un vídeo de YouTube")
st.write("Pega la URL de un vídeo para descargar sus comentarios y analizarlos en lote.")

video_url = st.text_input("URL del vídeo de YouTube:")
limite_comentarios = st.slider("Número de comentarios a analizar:", min_value=5, max_value=100, value=20)

if st.button("Analizar comentarios del vídeo"):
    if video_url.strip() == "":
        st.warning("Por favor, pega una URL de YouTube.")
    else:
        with st.spinner("Descargando y analizando comentarios..."):
            try:
                resultados = analyze_youtube_video(video_url, ensemble, tfidf, limit=limite_comentarios)
            except ValueError:
                st.warning("No se pudo reconocer un ID de vídeo válido en esa URL.")
            except Exception as e:
                st.error(f"No se pudieron descargar los comentarios: {e}")
            else:
                if not resultados:
                    st.info("El vídeo no tiene comentarios disponibles.")
                else:
                    total = len(resultados)
                    toxicos = sum(1 for r in resultados if r["es_toxico"])
                    st.write(f"**{toxicos} de {total}** comentarios analizados son tóxicos.")

                    tabla = pd.DataFrame(resultados)[["comentario", "es_toxico", "votos_toxico", "total_modelos"]]
                    tabla["es_toxico"] = tabla["es_toxico"].map({True: "⚠️ Tóxico", False: "✅ No tóxico"})
                    tabla = tabla.rename(columns={
                        "comentario": "Comentario",
                        "es_toxico": "Resultado",
                        "votos_toxico": "Votos tóxico",
                        "total_modelos": "Total modelos",
                    })
                    st.dataframe(tabla, use_container_width=True)
