# 📒 Apuntes del proyecto — Asistente RAG Retail (LlamaIndex + LangGraph)

> Notas de aprendizaje mientras montaba el entorno y entendía el proyecto.
> Pensado para repasar y para explicarlo en entrevistas (pivote a Data/AI Engineer).

---

## 🎯 De qué va el proyecto

Un **asistente de atención al cliente** para una tienda tipo retail (estilo Carrefour).
Responde preguntas sobre un catálogo de productos y servicios. Tiene dos niveles:

- **`1_rag.py`** → un **RAG**: busca en los documentos de `data/` y responde citando la fuente.
- **`2_agente.py`** → un **agente**: decide entre varias herramientas (el RAG + una calculadora de envío) y sabe encadenarlas.

**RAG vs Agente (pregunta típica de entrevista):**
- El **RAG** solo *recupera-y-responde*. Una única capacidad.
- El **agente** *decide* qué hacer en cada turno y puede usar varias herramientas (incluido el propio RAG). Patrón **ReAct** (Reason + Act).

**Los 4 pasos del RAG:**
1. Cargar documentos (leer la carpeta `data/`)
2. Trocear + generar *embeddings* (cada trozo → un vector numérico)
3. Indexar (guardar los vectores en un *vector store*)
4. Consultar (la pregunta → vector, se buscan los trozos más parecidos, el LLM redacta)

---

## 🧠 ¿Qué es RAG? (Retrieval-Augmented Generation)

Técnica para que un LLM (Gemini) responda **basándose en TUS documentos**, no solo en lo
que aprendió al entrenarse.

**Problema que resuelve:**
1. El LLM **no conoce tus datos privados** (precios, políticas, horarios de tu tienda).
2. El LLM **se inventa cosas ("alucina")** cuando no sabe algo.
RAG soluciona ambos: le *das* la info correcta antes de responder y le pides que use solo esa.

**El nombre = los 3 pasos:**
- **Retrieval** (Recuperación) → *buscas* en tus documentos los trozos relevantes.
- **Augmented** (Aumentada) → *añades* esos trozos a la pregunta que envías al modelo.
- **Generation** (Generación) → el modelo *genera* la respuesta usando esos trozos.

**Analogía del examen:**
- LLM sin RAG = examen de memoria (si no lo sabe, improvisa/inventa).
- LLM con RAG = examen con los apuntes delante (busca la página, la lee y responde de ahí).

**El truco técnico — los *embeddings*:**
La búsqueda no es por palabras exactas, sino por **significado**. Cada trozo de documento
y también la pregunta se convierten en un **vector** (lista de números que representa su
significado). Se buscan los trozos cuyo vector es más *parecido* al de la pregunta. Así
"¿se puede devolver marisco?" encuentra un texto que dice "productos frescos" aunque no
coincidan las palabras. En este proyecto los genera `llama-index-embeddings-gemini`.

**Por qué importa (Data/AI Engineer):** RAG es la técnica estrella para usar LLMs en
empresas (chatbots de soporte, búsqueda sobre documentación interna...). La oferta de
Carrefour lo pide (lo llaman ==*Vertex AI Search*==, la versión de Google de esto).

---

## 📦 Para qué sirve cada librería

### RAG — LlamaIndex
- **`llama-index`** → El *framework* principal de RAG. Se encarga de todo el flujo: cargar
  documentos, trocearlos, crear el índice de vectores y responder consultas recuperando
  los fragmentos relevantes. Es "el cerebro" del `1_rag.py`.
- **`llama-index-llms-gemini`** → El *conector* entre LlamaIndex y el modelo **Gemini** de
  Google. Permite que LlamaIndex use Gemini como LLM para redactar las respuestas.
- **`llama-index-embeddings-gemini`** → El conector para generar los **embeddings** (los
  vectores numéricos) usando el modelo de embeddings de Google. Es lo que convierte el
  texto en números para poder "buscar por similitud".

### Agente — LangGraph + LangChain
- **`langgraph`** → El *framework* para construir **agentes** como un **grafo** con ciclos
  (razonar → usar herramienta → volver a razonar). Aquí crea el agente estilo ReAct con
  `create_react_agent`. Es "el cerebro" del `2_agente.py`.
- **`langchain`** → La librería base del ecosistema LangChain. Aporta las piezas comunes,
  entre ellas el decorador **`@tool`** que convierte una función normal de Python en una
  "herramienta" que el agente puede decidir usar.
- **`langchain-google-genai`** → El *conector* entre LangChain/LangGraph y Gemini. Aporta
  `ChatGoogleGenerativeAI`, el objeto que representa el modelo Gemini para el agente.

### Utilidades
- **`python-dotenv`** → Lee el archivo **`.env`** y carga sus variables (como la clave de
  API) al entorno. Es lo que hace `load_dotenv()` al principio de los scripts. Sirve para
  no escribir claves secretas dentro del código.
- **`google-generativeai`** (usada en `listar_modelos.py`) → La librería oficial de Google
  para hablar con Gemini directamente. Aquí solo se usa como diagnóstico: listar qué
  modelos admite tu clave.

> 💡 Patrón que se repite: hay una **librería principal** (LlamaIndex / LangGraph) y un
> **conector** específico para cada proveedor de LLM (aquí, Gemini de Google). Si mañana
> quisieras usar OpenAI en vez de Gemini, cambiarías el conector, no todo el código.

### Otros LLM que se pueden usar con LlamaIndex (no solo Gemini)
LlamaIndex soporta 70+ integraciones. Todas siguen el patrón **`llama-index-llms-<proveedor>`**:
- Nube (de pago): OpenAI (`...-openai`), Anthropic/Claude (`...-anthropic`), Google/Gemini
  (`...-gemini`), Azure OpenAI, AWS Bedrock.
- Locales/gratis (corren en tu Mac, sin cuota): **Ollama** (`...-ollama`) ← útil para evitar
  los `limit: 0` de Gemini, HuggingFace, LlamaCPP, vLLM.
- Otros: Cohere, Mistral, Groq, Together.ai, Fireworks, Replicate, OpenRouter, LiteLLM...

**Cambiar de LLM = 3 pasos** (el resto del RAG no cambia):
1. `pip install llama-index-llms-openai`
2. `from llama_index.llms.openai import OpenAI`
3. `Settings.llm = OpenAI(...)`

**Cómo verificarlo (no memorizar, saber dónde mirar):**
- Docs oficiales: https://developers.llamaindex.ai/python/framework/module_guides/models/llms/modules/
- LlamaHub (directorio de integraciones): https://llamahub.ai
- PyPI: buscar `llama-index-llms` en https://pypi.org
- Truco: `pip install llama-index-llms-<X>` → si existe, se instala.
- ⚠️ Verificar siempre en la fuente oficial (las listas cambian; ni la memoria de una IA vale).

---

## 🔍 Los `import` de `1_rag.py`, uno a uno

Ojo: no todos son *librerías* enteras; la mayoría son **componentes concretos** (clases o
funciones) que se sacan de una librería.

### Utilidades básicas
- **`import os`** → Módulo estándar para hablar con el sistema operativo. Aquí lee las
  **variables de entorno** (`os.environ` / `os.getenv`): la clave de API y el modelo.
- **`from dotenv import load_dotenv`** → Función que **lee el `.env`** y carga sus variables
  al entorno. Sin ella, `os.environ["GOOGLE_API_KEY"]` no encontraría la clave.
  (Flujo: `load_dotenv()` carga → `os` lee.)

> 🔑 **¿De dónde lee `os` exactamente?** NO lee el archivo `.env`. `os` lee el **entorno en
> memoria** del programa (`os.environ`). La cadena es:
> ```
> .env (disco) --load_dotenv()--> os.environ (memoria) --os.getenv()--> tu variable
> ```
> `.env` es un fichero en disco; `os.environ` es una "pizarra" en memoria. `load_dotenv()`
> es el puente que copia lo uno en lo otro. Por eso el orden importa: primero `load_dotenv()`,
> luego `os.environ[...]`. La pizarra también puede llenarse con `export VAR=...` en la
> terminal o por el sistema operativo; `os` no distingue de dónde vino cada variable.

### El núcleo de LlamaIndex (`from llama_index.core import ...`)
- **`SimpleDirectoryReader`** → **Lee los documentos** de una carpeta (`data/`). → Paso 1: Cargar.
- **`VectorStoreIndex`** → Construye el **índice de vectores**: trocea, genera *embeddings* e
  indexa. → Pasos 2 y 3: Trocear + Indexar.
- **`StorageContext`** → Gestiona **guardar/cargar el índice en disco** (`storage/`) para no
  recalcularlo cada vez. → Persistencia.
- **`load_index_from_storage`** → **Carga** un índice ya guardado (rápido, no gasta API). → Persistencia.
- **`Settings`** → Configuración **global**: qué LLM y qué modelo de *embeddings* usar en todo el proceso.

### Los conectores con Gemini
- **`from llama_index.llms.gemini import Gemini`** → El **modelo de lenguaje** Gemini dentro
  de LlamaIndex. Es el que **redacta las respuestas**. Se asigna con `Settings.llm = Gemini(...)`.
- **`from llama_index.embeddings.gemini import GeminiEmbedding`** → El modelo de **embeddings**
  de Google. **Convierte texto en vectores** (documentos y pregunta). Se asigna con
  `Settings.embed_model = GeminiEmbedding(...)`.

> Son **dos modelos distintos**: `GeminiEmbedding` "traduce texto a números" (para buscar por
> significado); `Gemini` "escribe la respuesta final" (en lenguaje humano).

---

## ❓ Dudas que tuve (y sus respuestas)

### Entorno y herramientas base

**¿Qué es Homebrew?**
El "instalador de programas" de la terminal para Mac (como una *app store* de herramientas
de desarrollo). Con `brew install X` descarga, instala y gestiona programas y sus
dependencias desde un único sitio. Se instala **globalmente** (para todo el Mac), no por
proyecto.

**Problema que tuve: `Bad CPU type in executable`.**
Mi Mac es **Apple Silicon (M4, arm64)** pero tenía instalado el Homebrew de **Intel**
(migrado de mi Mac antiguo), y además sin Rosetta 2. Solución: **borrar** el Homebrew de
Intel (estaba vacío, no perdí nada) e **instalar el Homebrew nativo** de Apple Silicon,
que vive en `/opt/homebrew/` (el de Intel vivía en `/usr/local/`).

**¿Qué implica el `zsh` en `eval "$(brew shellenv zsh)"`?**
`brew shellenv` no instala nada: **imprime instrucciones** para añadir Homebrew al **PATH**
(la lista de carpetas donde la terminal busca programas). El `eval` las ejecuta. El `zsh`
le dice que las escriba en la *sintaxis de zsh* (mi shell). Sin argumento, lo detecta solo.
Al ponerlo en el `.zprofile`, se aplica automáticamente en cada terminal nueva.

**¿Cómo comprobar qué versiones de Python tengo instaladas? (manualmente)**
- `which python3` y `python3 --version` → cuál está activo y de dónde sale.
- Probar `python3.10 --version`, `python3.11`, `python3.12`... → las que existan responden.
- `ls /usr/bin/python*`, `/usr/local/bin/`, `/opt/homebrew/bin/` → binarios físicos.
- `brew list --versions | grep python`, `pyenv versions`, `conda --version` → por gestor.
- `find ~/Documents -name pyvenv.cfg` → entornos virtuales existentes.

### Entornos virtuales (venv)

**¿`brew install python@3.12` instala Python solo para este proyecto?**
No. Homebrew instala **globalmente**, para todos los proyectos. La carpeta desde la que
ejecutes el comando da igual.

**¿Por qué aislar las librerías por proyecto?**
1. **Sin conflictos de versiones** (proyecto A con langchain 0.3, B con 0.1, sin pelearse).
2. **Reproducibilidad** ("en mi máquina funciona" → el `requirements.txt` recrea el entorno).
3. **No ensucias el Python del sistema**.
4. **Limpieza fácil** (`rm -rf .venv` y desaparece todo).
5. **Sabes de qué depende tu proyecto** (el entorno empieza vacío).

**¿La gente usa venv-en-el-proyecto o Anaconda?**
Ambos existen. **Data Science** → suele usar `conda` (entornos en sitio central).
**Desarrollo / Data Engineering** → suele usar **`venv` dentro del proyecto** + Docker.
Para mi objetivo (Data Engineer) → `venv` (y en el futuro, `uv`, que es más rápido).

**¿Qué hace `python3.12 -m venv .venv`?**
Crea una carpeta `.venv/` con un Python 3.12 **aislado y vacío** para este proyecto.
`-m venv` ejecuta el módulo `venv` (viene con Python). `.venv` es el nombre estándar (el
punto la hace oculta). Se ejecuta **una sola vez** por proyecto. No descarga Python de
nuevo: crea un enlace ligero al Python 3.12 ya instalado.

**No veía la carpeta `.venv`.**
Porque empieza por punto → está **oculta**. Se ve con `ls -la` (la `-a` = mostrar ocultos)
o en Finder con `Cmd + Shift + .`.

**¿Qué hace `source .venv/bin/activate`?**
"Enciende" el entorno en la terminal actual. `source` ejecuta el script *dentro de* mi
sesión (para que los cambios se queden). Efecto: `python` y `pip` pasan a ser los del
`.venv`, aparece `(.venv)` en el prompt, y se puede salir con `deactivate`.

**¿Qué es "la sesión actual"?**
La ventana/pestaña de terminal abierta ahora mismo, con su memoria temporal (carpeta,
variables, entorno activado). Cada ventana es una sesión independiente. Al cerrarla se
olvida lo temporal → por eso hay que reactivar el `.venv` en cada terminal nueva.

**¿Por qué hay que activarlo dentro de cada sesión?**
Porque activar = decir "en qué proyecto trabajo AHORA, en esta ventana". Es una decisión
momentánea, no una configuración fija. Así puedo tener varias ventanas con proyectos
distintos a la vez sin que se pisen. Homebrew sí es permanente (va en `.zprofile`) porque
es una herramienta base; el entorno de un proyecto se activa a mano a propósito.

**Una vez activado, ¿las librerías que instale con pip van a `.venv`?**
Sí. Con `(.venv)` activo, `pip install` instala en `.venv/lib/python3.12/site-packages/`
(la "despensa" del proyecto), no en el Python global. Comprobar con `which pip` / `pip -V`.
Regla: **antes de `pip install`, mira que veas `(.venv)` en el prompt.**

### Configuración (.env)

**⚠️ `.venv` (carpeta del entorno) vs `.env` (archivo de config): NO son lo mismo.**
- `.venv` = carpeta con Python + librerías. **No se toca a mano.**
- `.env` = archivo de texto con la clave secreta. **Sí hay que editarlo** para poner la clave.

**¿Qué hace `cp .env.example .env`?**
`cp ORIGEN DESTINO` = copiar. Hace una copia de la plantilla `.env.example` y la llama
`.env`. La plantilla (pública, se sube a GitHub) muestra QUÉ variables hacen falta sin
valores reales; la copia `.env` (privada, va en `.gitignore`) es donde pongo mi clave real.
Es solo la mitad: después hay que **editar `.env`** y reemplazar `pega_aqui_tu_clave` por
la clave de verdad.

**⚠️ Ojo con la clave:** las de Google AI Studio suelen empezar por `AIza...`. Conseguir
gratis en https://aistudio.google.com/app/apikey. No compartirla ni subirla a GitHub.

---

## ✅ Checklist de puesta en marcha

- [x] Homebrew nativo (Apple Silicon) instalado y en el PATH (`.zprofile`)
- [x] Python 3.12 instalado (`brew install python@3.12`)
- [x] Entorno virtual creado (`python3.12 -m venv .venv`)
- [x] Entorno activado (`source .venv/bin/activate` → ver `(.venv)`)
- [x] Dependencias instaladas (`pip install -r requirements.txt`)
- [x] Clave de Gemini en `.env` (la clave válida empieza por `AQ.`, no siempre `AIza`)
- [x] Probar el RAG: `python 1_rag.py` ✅ FUNCIONA
- [ ] Probar el agente: `python 2_agente.py`

### ⚙️ Configuración de modelos que FUNCIONA (plan gratuito, cuenta nueva)
```
GEMINI_MODEL=models/gemini-flash-lite-latest     # texto (LLM)
GEMINI_EMBED_MODEL=models/gemini-embedding-001   # embeddings
```

### 🐞 Problemas resueltos al ejecutar (por si vuelven)
1. **`text-embedding-004 not found`** → el modelo de embeddings del `.env` estaba obsoleto.
   Solución: `gemini-embedding-001`. (Un typo `GEMINI_pythonEMBED_MODEL` hizo que se usara
   el valor por defecto obsoleto → lección: si un cambio "no hace efecto", revisa el nombre
   de la variable.)
2. **`429 RESOURCE_EXHAUSTED ... limit: 0` con `gemini-2.0-flash`** → ese modelo tiene cuota
   0 en el plan gratuito de esta cuenta. Solución: usar un modelo con cuota (Flash Lite).
3. **`404 ... no longer available to new users` con `gemini-2.5-flash-lite`** → el nombre a
   secas está bloqueado para cuentas nuevas. Solución: usar el alias `-latest`
   (`gemini-flash-lite-latest`).
4. Regla: al cambiar el **modelo de embeddings** hay que **borrar `storage/`** (reindexar);
   al cambiar solo el de **texto**, NO hace falta.
5. Los `FutureWarning` / `DeprecationWarning` de `google.generativeai` y `Gemini`/
   `GeminiEmbedding` son solo avisos (librerías antiguas), no detienen el programa.

---

## 🔁 Recordatorio para volver al proyecto otro día

```bash
cd ".../asistente-rag-retail (LlamaIndex + LangGraph)"
source .venv/bin/activate      # reactivar el entorno (ver "(.venv)" en el prompt)
python 1_rag.py                # o 2_agente.py
```
