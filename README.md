# Pre-entrega 4: Sistema RAG escalable en la nube con Pinecone

Curso AI Engineer (Coderhouse), Módulo 4: Escalabilidad documental, RAG avanzado y Pinecone.

Este repo es un **Módulo de Recuperación Escalable**. Toma un conjunto de documentos técnicos (Markdown, JSON o PDF), los sube a un índice **Pinecone Serverless** con metadatos, los recupera con un **recuperador híbrido** (BM25 léxico + búsqueda vectorial, combinados con `EnsembleRetriever`) y mide la calidad de la recuperación con **Precision@5 y Recall@5** sobre un Golden Set.

```
                ┌──────────────── ingest.py ────────────────┐
data/docs/ ───> │ carga → limpieza → chunking (600 tokens)  │ ──batch upsert──> Pinecone Serverless
 .md .json .pdf │ → metadatos → embeddings (1536 dims)      │                   índice: python-docs-rag
                └───────────────────────────────────────────┘                   namespace: python-stdlib-es
                                                                                (texto original en metadata)
                ┌──────────── RAGSystem (app/rag_system.py) ──────────────┐            │
consulta ─────> │ BM25Retriever (léxico)       ── top-10 ─┐               │ <── corpus ┤
                │ PineconeRetriever (semántico) ── top-10 ─┴─> Ensemble   │ <── query ─┘
                │                                     (RRF ponderado)     │
                └──────────────────────────────────────────── top-5 ──────┘ ──> evaluate.py: Recall@5 / Precision@5
```

## Estructura

```
.
├── app/
│   ├── config.py          # Configuración única: índice, namespace, dimensión, chunking, top-k, pesos
│   ├── embeddings.py      # Modelo de embeddings (Google u OpenAI, 1536 dims) + control de dimensión
│   ├── pinecone_store.py  # SDK de Pinecone: crear/verificar el índice serverless, listar y leer el corpus
│   ├── documents.py       # Loaders (md/json/pdf), limpieza, chunking y esquema de metadatos
│   ├── retrievers.py      # PineconeRetriever (LangChain) + BM25Retriever con tokenizador técnico
│   ├── rag_system.py      # Clase RAGSystem: EnsembleRetriever (BM25 + Pinecone)
│   └── metrics.py         # Recall@k, Precision@k, Hit Rate, MRR, F1
├── data/
│   ├── docs/              # Dataset: documentación (en español) de la librería estándar de Python
│   └── golden_set.json    # Benchmark: 10 preguntas con su documento esperado
├── setup_index.py         # Paso 1: verifica/crea el índice serverless
├── ingest.py              # Paso 2: pipeline de ingesta a Pinecone
├── query.py               # Consultas manuales (interactivo o una sola consulta)
├── evaluate.py            # Paso 3: Recall@5 y Precision@5 sobre el Golden Set
├── tests/                 # Tests offline (no usan API keys)
├── requirements.txt
└── .env.example
```

### Dataset

`data/docs/` tiene documentación técnica en español de 8 módulos de la librería estándar de Python (`asyncio`, `concurrent.futures`, `functools`, `dataclasses`, `pathlib`, `logging`, `json`, `typing`) en Markdown, más un FAQ de errores comunes en **JSON** (8 entradas). El loader también acepta **PDF** (una página = una `page` en la metadata): alcanza con copiar un `.pdf` a `data/docs/` y volver a correr la ingesta.

Algunos documentos se pisan a propósito (asyncio y concurrent.futures hablan los dos de concurrencia, y el FAQ menciona asyncio, dataclasses y lru_cache). Así la evaluación tiene "distractores" realistas y no resulta trivial.

## Cómo replicar el índice de Pinecone

### 1. Instalación

Probado con Python 3.14 (funciona con 3.11 o superior).

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
source .venv/bin/activate         # Linux / Mac
pip install -r requirements.txt
```

### 2. Variables de entorno

```bash
cp .env.example .env              # Windows: copy .env.example .env
```

Completá en `.env`:

| Variable | Descripción |
|---|---|
| `PINECONE_API_KEY` | API key de Pinecone (plan gratuito en <https://app.pinecone.io>, menú *API Keys*) |
| `INDEX_NAME` | Nombre del índice (default `python-docs-rag`) |
| `PINECONE_NAMESPACE` | Namespace donde se guardan estos documentos (default `python-stdlib-es`) |
| `EMBEDDING_PROVIDER` | `google` (default, tier gratuito) u `openai` |
| `GOOGLE_API_KEY` / `OPENAI_API_KEY` | La key del proveedor elegido |

`.env` está en `.gitignore`. Las claves nunca se escriben en el código: `app/config.py` las lee del entorno y, si falta alguna, el programa corta enseguida con un mensaje que dice cuál es.

### 3. Crear el índice serverless

```bash
python setup_index.py
```

El script verifica si el índice existe. Si no existe, lo crea con `ServerlessSpec(cloud="aws", region="us-east-1")`, **dimensión 1536** y **métrica coseno**, y espera a que esté listo. Si ya existe, comprueba que la dimensión y la métrica coincidan con la configuración y, si no coinciden, frena con un error claro (*mismatch de dimensiones*). Se puede correr las veces que haga falta.

### 4. Ingestar los documentos

```bash
python ingest.py --dry-run   # opcional: muestra chunks y metadata sin llamar a ninguna API
python ingest.py             # sube a Pinecone solo los documentos nuevos o modificados
python ingest.py --force     # vacía el namespace y reindexa todo
```

Con el dataset incluido se generan **32 chunks** a partir de 9 documentos: 3 por cada `.md` (la mayoría entre 400 y 590 tokens; el último de cada documento puede ser más corto) y 1 por cada entrada del FAQ.

### 5. Consultar y evaluar

```bash
python query.py "¿Cómo limito la concurrencia en asyncio?"
python query.py --modo bm25 "ensure_ascii"              # un solo retriever, para comparar
python query.py --categoria concurrencia "hilos o procesos"   # hard filter por metadata
python evaluate.py                                      # Recall@5 / Precision@5
python -m pytest -q                                     # tests offline
```

## Decisiones de diseño

### Infraestructura (Pinecone)

- **Una sola fuente de verdad para la dimensión.** `EMBEDDING_DIM=1536` vive en `app/config.py`. Con ese valor se crea el índice y se le pide al modelo el tamaño del vector (`output_dimensionality` en Gemini, `dimensions` en OpenAI). Además, cada vector se valida antes del upsert.
- **Embeddings intercambiables.** Por defecto se usa `gemini-embedding-001` porque tiene tier gratuito, reducido a 1536 dims (el modelo es Matryoshka, así que se puede truncar sin reentrenar). Con `EMBEDDING_PROVIDER=openai` se usa `text-embedding-3-small`, también de 1536, así que **el mismo índice sirve para los dos**. Ojo: si cambiás de proveedor, reindexá con `--force`, porque los vectores de modelos distintos no son comparables entre sí.
- **Métrica coseno**, que es la métrica para la que están optimizados ambos modelos.
- **Namespace** `python-stdlib-es`. Ingesta y consulta usan siempre el mismo namespace, así este corpus queda aislado de otros datasets o entornos (dev/prod) que compartan el índice.
- **SDK nativo de Pinecone + retriever propio de LangChain.** `langchain-pinecone` todavía no publica versiones para Python 3.14. Por eso `PineconeRetriever` (en `app/retrievers.py`) es un `BaseRetriever` de LangChain armado sobre `index.query(...)`, con soporte de namespace y de filtros de metadata. Para el `EnsembleRetriever` es un retriever más.

### Ingesta y metadatos

- **El texto original va en la metadata** (`metadata["text"]`). Una sola consulta a Pinecone devuelve vector, texto y fuente, sin pasar por una base relacional.
- **Esquema de metadatos único** (`ChunkMetadata` en `app/documents.py`), igual para Markdown, JSON y PDF, para evitar *schema drift*:

  | Campo | Ejemplo | Para qué sirve |
  |---|---|---|
  | `text` | "## Limitar la concurrencia..." | Contenido del chunk |
  | `chunk_id` | `asyncio#p1-c2` | ID del vector (determinístico) |
  | `doc_id`, `source`, `title` | `asyncio`, `asyncio.md` | Trazabilidad y evaluación |
  | `section` | "Limitar la concurrencia con Semaphore" | Encabezado Markdown del chunk |
  | `page` | `1` / nº de entrada JSON / nº de página PDF | Cita de la fuente |
  | `category`, `tags`, `library` | `concurrencia`, `["async", ...]` | Filtros (`$eq`, `$in`) |
  | `doc_type`, `language` | `markdown`, `es` | Filtros por formato o idioma |
  | `chunk_index`, `n_tokens` | `2`, `400` | Diagnóstico del chunking |
  | `content_hash`, `ingested_at` | `2bf83a0d...`, `2026-10-07` | Reindexación incremental |

- **Limpieza antes de fragmentar.** Se normaliza Unicode (NFC), comillas, espacios y saltos de línea, se eliminan caracteres de control y se deduplican párrafos repetidos (por ejemplo, encabezados de PDF que se repiten en cada hoja).
- **Chunking.** Se usa `RecursiveCharacterTextSplitter` con **600 tokens y 100 de overlap**, medidos con `tiktoken` y no en caracteres (el rango sugerido es 500 a 800). Los separadores priorizan la estructura Markdown (`## `, bloques de código, párrafos), así que cada chunk suele arrancar en una sección.
- **Encabezado contextual.** Lo que se embebe (y lo que indexa BM25) es `título — sección` seguido del chunk. Un fragmento como "Se usa como decorador..." no dice de qué librería habla; con el encabezado, sí. En la metadata se guarda solo el texto original.
- **Batch upsert** de 100 vectores por request, dentro del rango de 100 a 200 recomendado.
- **Ingesta idempotente e incremental.** Los IDs son determinísticos (`<doc_id>#p<página>-c<n>`) y cada chunk guarda el hash de su documento. Si un documento no cambió, se saltea (no se vuelven a pagar sus embeddings). Si cambió, se borran sus chunks viejos por prefijo y se suben los nuevos. Correr la ingesta dos veces no duplica vectores.

### Recuperador híbrido (`RAGSystem`)

- **BM25** (`BM25Retriever` de LangChain) usa un tokenizador pensado para documentación técnica en español: pasa todo a minúsculas, saca acentos y stopwords, conserva identificadores completos (`ensure_ascii`) y agrega sus partes (`ensure`, `ascii`), y separa nombres con punto (`json.dumps`). BM25 es el que encuentra términos exactos como `default_factory`, `PYTHONASYNCIODEBUG` o mensajes de error literales.
- **Pinecone** cubre las paráfrasis. Por ejemplo, "¿cómo guardo el resultado de una función costosa?" no tiene ninguna palabra en común con `lru_cache`, pero el embedding las relaciona.
- **El corpus de BM25 sale del propio Pinecone.** Al iniciar, `RAGSystem.from_pinecone()` recorre el namespace (`index.list` + `index.fetch`) y arma BM25 con el `text` de la metadata. Los dos retrievers ven exactamente los mismos chunks con los mismos IDs, y no hace falta tener los archivos locales para consultar.
- **Fusión con `EnsembleRetriever`**, que aplica *Reciprocal Rank Fusion* ponderado: `score(d) = Σ wᵢ / (60 + rankᵢ(d))`. Combina **posiciones**, no puntajes, porque un score BM25 de 12.3 y una similitud coseno de 0.81 están en escalas distintas y no se pueden sumar directamente. Se deduplica por `chunk_id` (`id_key`), así un chunk que encuentran los dos retrievers suma ambos aportes y sube en el ranking.
- **Más candidatos que el top-k.** Cada retriever aporta 10 candidatos y después del RRF se corta en 5, así la fusión tiene material para reordenar.
- **Filtros de metadata.** `rag.retrieve(q, metadata_filter={"category": "concurrencia"})` aplica el filtro como *hard filter* en Pinecone, y el mismo filtro en memoria sobre BM25.
- Los pesos `[BM25, vectorial]` se configuran (`BM25_WEIGHT` / `VECTOR_WEIGHT`, default 0.5/0.5), y `mode="bm25" | "vector"` permite aislar cada retriever.

## Evaluación (`evaluate.py`)

El Golden Set (`data/golden_set.json`) tiene **10 preguntas** (la consigna pide 5) con su `documento_id_esperado`, que puede ser uno o una lista. Mezcla preguntas **léxicas** (términos exactos), **semánticas** (paráfrasis sin las palabras clave) y **mixtas**.

Definiciones (en `app/metrics.py`):

- **Recall@5**: documentos relevantes encontrados en el top-5 sobre el total de documentos relevantes. Con un solo documento esperado responde a la pregunta "¿está el documento correcto entre los 5?" y vale 0 o 1.
- **Precision@5**: chunks del top-5 que pertenecen a un documento relevante, dividido por 5. Responde "¿qué porcentaje de los 5 recuperados es útil?".
- Además se calculan **Hit Rate**, **MRR** (qué tan arriba aparece el primer acierto) y **F1**.
- **Techo de Precision@5.** Si los documentos relevantes tienen en total 3 chunks, ni un retriever perfecto supera 3/5 = 60%. El script informa ese techo para poder leer bien la precisión obtenida.

`python evaluate.py` imprime el detalle por pregunta del modo híbrido y una tabla comparativa de **Híbrido vs. solo BM25 vs. solo Pinecone**:

```
========================================================================================
RESUMEN — 10 preguntas del Golden Set, top-5, índice 'python-docs-rag', namespace 'python-stdlib-es'
========================================================================================
Modo                          Recall@5   Precision@5  Hit Rate     MRR      F1
------------------------------------------------------------------------------
Híbrido (BM25 + Pinecone)          ...           ...       ...     ...     ...
Solo BM25                        95.0%         52.0%    100.0%   1.000   0.672
Solo Pinecone                      ...           ...       ...     ...     ...
------------------------------------------------------------------------------
Techo teórico de Precision@5 con este Golden Set: 68.0% (los documentos relevantes no siempre tienen 5 chunks).
```

### Resultados

> La fila **Solo BM25** se midió offline (BM25 no necesita API keys). Las filas **Híbrido** y **Solo Pinecone** se completan corriendo `python evaluate.py` con el índice ya cargado.

| Modo | Recall@5 | Precision@5 | Hit Rate | MRR | F1 |
|---|---|---|---|---|---|
| Híbrido (BM25 + Pinecone) | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ |
| Solo BM25 | 95.0% | 52.0% | 100.0% | 1.000 | 0.672 |
| Solo Pinecone | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ |

Techo teórico de Precision@5: **68%**.

El único fallo de BM25 está en la pregunta de los argumentos por defecto mutables: encuentra el FAQ pero no el documento de `dataclasses` (Recall 50%). Esa respuesta depende del significado ("valor por defecto" ≈ "default_factory") y no de las palabras exactas, que es justamente el caso que la parte vectorial del híbrido tiene que cubrir.

## Tests

```bash
python -m pytest -q
```

Son 12 tests que corren **sin red ni API keys**. Usan un retriever vectorial simulado y cubren:

- el esquema de metadatos (todos los chunks con los mismos campos),
- el límite de tokens por chunk y los IDs únicos y determinísticos,
- los loaders Markdown y JSON y la detección de secciones,
- el tokenizador de BM25 y los filtros de metadata,
- la fusión RRF (un chunk que encuentran ambos retrievers queda primero, sin duplicados),
- las fórmulas de las métricas.
