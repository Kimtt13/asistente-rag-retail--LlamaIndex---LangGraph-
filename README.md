# Asistente de tienda con RAG y agentes (LlamaIndex + LangGraph + Gemini)

Un asistente que responde preguntas sobre un
catálogo de productos y servicios de una tienda
usando **RAG** con **LlamaIndex** y un **agente** con **LangGraph**, sobre el
LLM **Gemini** de Google.

---

## Qué hace

- **`1_rag.py`** — Un RAG que busca en los documentos de `data/` (fichas de
  producto, política de devoluciones, horarios) y responde citando la fuente.
- **`2_agente.py`** — Un agente LangGraph que decide entre dos herramientas:
  el RAG del catálogo y una calculadora de coste de envío. Sabe encadenarlas.


