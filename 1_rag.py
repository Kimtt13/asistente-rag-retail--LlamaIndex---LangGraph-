"""
====================================================================
DÍA 1 (SÁBADO) — RAG básico con LlamaIndex + Gemini
====================================================================

QUÉ ES ESTO
-----------
RAG = Retrieval-Augmented Generation. En lugar de que el LLM responda
"de memoria" (y se invente cosas), primero BUSCAMOS los fragmentos
relevantes en NUESTROS documentos (el catálogo de la tienda) y se los
damos al modelo para que responda basándose SOLO en ellos.

FLUJO DEL RAG (los 4 pasos clásicos):
  1. Cargar documentos      -> SimpleDirectoryReader lee la carpeta data/
  2. Trocear + embeddings    -> cada trozo se convierte en un vector numérico
  3. Indexar                 -> VectorStoreIndex guarda esos vectores
  4. Consultar               -> la pregunta se convierte en vector, se buscan
                                los trozos más parecidos y el LLM redacta

Ejecuta:  python 1_rag.py
"""
import os
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# --- Configuración ---------------------------------------------------
load_dotenv()  # lee las variables del fichero .env

API_KEY = os.environ["GOOGLE_API_KEY"]
LLM_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")

DATA_DIR = "data"        # carpeta con los documentos de la tienda
STORAGE_DIR = "storage"  # aquí se guarda el índice para no recalcularlo cada vez

# LlamaIndex usa un objeto global "Settings" para saber qué LLM y qué
# modelo de embeddings emplear en todo el proceso.
Settings.llm = Gemini(model=LLM_MODEL, api_key=API_KEY)
Settings.embed_model = GeminiEmbedding(model_name=EMBED_MODEL, api_key=API_KEY)


def build_query_engine():
    """Construye (o carga) el índice y devuelve un motor de consultas.

    La primera vez calcula los embeddings y guarda el índice en storage/.
    Las siguientes veces lo carga de disco (más rápido y gasta menos API).
    """
    if os.path.exists(STORAGE_DIR):
        print("Cargando índice existente desde 'storage/'...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    else:
        print("Construyendo el índice por primera vez (calculando embeddings)...")
        documents = SimpleDirectoryReader(DATA_DIR).load_data() # lee todos los documentos de data/.
        index = VectorStoreIndex.from_documents(documents)  # trocea, genera embeddings (llamando a Gemini) y los indexa. Aquí es donde se gasta la API de embeddings.
        index.storage_context.persist(persist_dir=STORAGE_DIR) # guarda el índice en storage/ para no repetir todo la próxima vez.

    # recupera los 3 trozos más relevantes por pregunta.
    return index.as_query_engine(similarity_top_k=3)


def main():
    query_engine = build_query_engine() # objeto que sabe responder preguntas. Se hace una sola vez (no por cada pregunta).

    # Algunas preguntas de ejemplo para ver el RAG en acción.
    preguntas = [
        "¿Qué eslo que suele comprar más la gente?",
        "¿Es apto para veganos el aceite de oliva? ¿Y la bebida de avena?",
        "¿Cuánto cuesta el envío a domicilio y cuándo es gratis?",
        "¿Puedo devolver pescado fresco?",
    ]

    for pregunta in preguntas:
        print("\n" + "=" * 70)
        print("PREGUNTA:", pregunta)
        respuesta = query_engine.query(pregunta)
        print("\nRESPUESTA:", respuesta)

        # Trazabilidad: de qué documento salió la respuesta (clave en RAG serio).
        print("\nFUENTES USADAS:")
        for nodo in respuesta.source_nodes:
            fichero = nodo.metadata.get("file_name", "desconocido")
            print(f"   - {fichero} (relevancia: {nodo.score:.2f})")


if __name__ == "__main__":
    main()
