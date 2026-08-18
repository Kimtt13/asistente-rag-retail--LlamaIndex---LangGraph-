"""
====================================================================
DÍA 2 (DOMINGO) — Agente con LangGraph que USA el RAG como herramienta
====================================================================

QUÉ ES ESTO Y EN QUÉ SE DIFERENCIA DEL DÍA 1
--------------------------------------------
El RAG del Día 1 solo sabe hacer una cosa: buscar en el catálogo.
Un AGENTE es un LLM que DECIDE, en cada turno, qué herramienta usar
(o si contesta directamente). Aquí le damos DOS herramientas:

  1. buscar_en_catalogo  -> por dentro llama a nuestro RAG (LlamaIndex)
  2. calcular_coste_envio -> una función normal de Python (cálculo exacto)

El agente lee la pregunta del usuario y razona:
  "¿Esto va de un producto/política? -> uso el catálogo.
   ¿Me piden calcular un envío? -> uso la calculadora.
   ¿Necesito ambas? -> las encadeno."

Esto es exactamente la "orquestación de flujos complejos" y los
"agentes" que pide la oferta. LangGraph modela al agente como un GRAFO
(razonar -> usar herramienta -> volver a razonar), con ciclos.

Ejecuta:  python 2_agente.py
"""
import os #Leer variables de entorno (API_KEY, GEMINI_MODEL)
from dotenv import load_dotenv #Cargar variables de entorno desde .env

from langchain_core.tools import tool #convierte funciones en herramientas para el agente
from langchain_google_genai import ChatGoogleGenerativeAI # modelo LLM de Google Gemini
from langgraph.prebuilt import create_react_agent # construye un agente estilo ReAct (razonar + actuar) a partir de un LLM y herramientas

# Reutilizamos el motor RAG que construimos
from importlib import import_module
rag = import_module("1_rag")  # el fichero se llama "1_rag.py"

load_dotenv()
API_KEY = os.environ["GOOGLE_API_KEY"]
LLM_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")

# El motor de consultas del RAG (se construye una sola vez al arrancar).
query_engine = rag.build_query_engine()


# --- HERRAMIENTA 1: el RAG envuelto como "tool" del agente ------------
@tool
def buscar_en_catalogo(pregunta: str) -> str:
    """Busca información sobre productos, precios, alérgenos, política de
    devoluciones, envíos u horarios de la tienda. Úsala para cualquier
    duda sobre el catálogo o los servicios de la tienda."""
    return str(query_engine.query(pregunta))


# --- HERRAMIENTA 2: una función normal de Python ---------------------
@tool
def calcular_coste_envio(importe_pedido_euros: float) -> str:
    """Calcula el coste de envío a domicilio dado el importe del pedido en
    euros. El envío es gratis a partir de 50 €; si no, cuesta 4,90 €."""
    if importe_pedido_euros >= 50:
        return "El envío es GRATUITO (pedido de 50 € o más)."
    return "El envío cuesta 4,90 € (pedido inferior a 50 €)."


def main():
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=API_KEY)
    herramientas = [buscar_en_catalogo, calcular_coste_envio]

    # create_react_agent construye el grafo del agente estilo "ReAct"
    # (Reason + Act): el LLM razona, decide una herramienta, ve el
    # resultado y vuelve a razonar hasta tener la respuesta final.
    agente = create_react_agent(llm, herramientas)

    print("Asistente de tienda (agente LangGraph). Escribe 'salir' para terminar.\n")
    print("Ejemplos que combinan las dos herramientas:")
    print("  - ¿El café en cápsulas es compatible con Nespresso y cuánto cuesta?")
    print("  - Mi pedido es de 32 €, ¿cuánto pago de envío?")
    print("  - Quiero aceite de oliva y bebida de avena. Con eso, ¿me sale gratis el envío?\n")

    while True:
        pregunta = input("Tú: ").strip()
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break
        if not pregunta:
            continue

        # El agente recibe el historial de mensajes y devuelve otro historial.
        resultado = agente.invoke({"messages": [("user", pregunta)]})
        respuesta_final = resultado["messages"][-1].text
        print("\nAsistente:", respuesta_final, "\n")


if __name__ == "__main__":
    main()
