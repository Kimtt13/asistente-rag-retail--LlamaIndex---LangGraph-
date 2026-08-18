"""
Utilidad de diagnóstico: lista los modelos de Gemini disponibles con TU clave.
Úsalo si un script te da un error del tipo "model not found".

Ejecuta:  python listar_modelos.py
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Carga variables de entorno, GOOGLE_API_KEY desde el fichero .env y autentica con la API de Gemini
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# recorre todos los modelos; de los que sepan generar texto, imprime su nombre
print("Modelos que puedes usar para GENERAR texto (LLM):\n")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(" -", m.name)

# recorre todos los modelos; de los que sepan generar embedding (convertir texto en vectores), imprime su nombre
print("\nModelos que puedes usar para EMBEDDINGS (RAG):\n")
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(" -", m.name)

print("\nCopia el nombre que quieras en tu fichero .env (GEMINI_MODEL / GEMINI_EMBED_MODEL).")
