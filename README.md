# Asistente de tienda con RAG y agentes (LlamaIndex + LangGraph + Gemini)

Mini-proyecto de fin de semana: un asistente que responde preguntas sobre un
catálogo de productos y servicios de una tienda (estilo retail/Carrefour),
usando **RAG** con **LlamaIndex** y un **agente** con **LangGraph**, sobre el
LLM **Gemini** de Google.

**Por qué este proyecto:** cubre de forma real las tecnologías que pide la
oferta de AI Engineer de Carrefour — IA generativa, RAG (equivalente a *Vertex
AI Search*), orquestación de agentes (*flujos complejos*), y stack Google.

---

## Qué hace

- **`1_rag.py`** — Un RAG que busca en los documentos de `data/` (fichas de
  producto, política de devoluciones, horarios) y responde citando la fuente.
- **`2_agente.py`** — Un agente LangGraph que decide entre dos herramientas:
  el RAG del catálogo y una calculadora de coste de envío. Sabe encadenarlas.

---

## Puesta en marcha (una vez)

Tu Mac trae Python 3.9, pero conviene uno más nuevo para estas librerías.
Instala Python 3.12 con Homebrew (no toca tu Python del sistema):

```bash
brew install python@3.12
```

Luego, dentro de la carpeta del proyecto:

```bash
# 1. Crear un entorno virtual aislado
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Instalar las dependencias
pip install -r requirements.txt

# 3. Configurar la clave de Gemini
cp .env.example .env
#    -> abre .env y pega tu clave (gratis en https://aistudio.google.com/app/apikey)
```

> Cada vez que vuelvas al proyecto: `source .venv/bin/activate`.

---

## Plan del fin de semana

### Sábado — RAG (LlamaIndex)
1. Instalación y clave (pasos de arriba).
2. `python 1_rag.py` → ver el RAG respondiendo las 3 preguntas de ejemplo.
3. **Experimenta** para entenderlo (aquí está el aprendizaje real):
   - Añade una ficha de producto nueva en `data/` y borra la carpeta
     `storage/` para reindexar. Pregúntale por ese producto.
   - Cambia `similarity_top_k` en `1_rag.py` y observa cómo cambian las fuentes.
   - Haz una pregunta cuya respuesta NO esté en los documentos y mira qué hace.

### Domingo — Agente (LangGraph)
1. `python 2_agente.py` → hablar con el agente.
2. Prueba preguntas que obliguen a **elegir herramienta** y a **combinarlas**
   (hay ejemplos al arrancar el script).
3. **Experimenta:**
   - Añade una tercera herramienta (p. ej. `comprobar_stock(producto)` que
     devuelva un valor inventado) y observa cómo el agente la incorpora.
   - Cambia la descripción (docstring) de una herramienta y mira cómo afecta a
     cuándo el agente decide usarla. *(Esto es prompt engineering real.)*

### Si te sobra tiempo (opcional, +valor)
- Añade **observabilidad** con Phoenix o Langfuse (¡la oferta lo pide!) para
  ver las trazas de cada llamada al LLM y a las herramientas.
- Monta una interfaz sencilla con `streamlit` o `gradio`.
- Sube el repo a **GitHub** con este README (la oferta valora proyectos en Git).

---

## Cómo funciona por dentro (para la entrevista)

- **RAG (`1_rag.py`):** cargar documentos → trocear y generar *embeddings* →
  indexar en un *vector store* → en cada pregunta, recuperar los trozos más
  parecidos y pasárselos al LLM para que responda con trazabilidad de fuentes.
- **Agente (`2_agente.py`):** patrón **ReAct** (Reason + Act). LangGraph modela
  al agente como un grafo con ciclos: el LLM razona, elige una herramienta, lee
  el resultado y vuelve a razonar hasta la respuesta final.
- **Diferencia RAG vs agente:** el RAG solo recupera-y-responde; el agente
  *decide* qué hacer y puede usar varias herramientas (incluido el propio RAG).

---

## Resolución de problemas

- **`model not found` / error de modelo:** ejecuta `python listar_modelos.py`
  para ver los modelos disponibles con tu clave y actualiza `.env`.
- **Error de instalación con pip:** asegúrate de estar en el entorno virtual
  (`source .venv/bin/activate`) y con Python 3.12 (`python --version`).
- **Respuestas raras o vacías:** borra la carpeta `storage/` para forzar la
  reindexación de los documentos.

---

## Para el CV / LinkedIn (una vez hecho)

> *Asistente de atención al cliente con RAG y agentes: pipeline de
> Retrieval-Augmented Generation sobre un catálogo de productos con
> **LlamaIndex**, orquestado como agente multi-herramienta con **LangGraph**
> sobre **Gemini (Google Cloud)**. Incluye recuperación con trazabilidad de
> fuentes y selección dinámica de herramientas.*

Con esto ya puedes afirmar con verdad que **has usado LlamaIndex y LangGraph**.
