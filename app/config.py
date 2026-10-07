"""
Configuración centralizada del Módulo de Recuperación Escalable.

Todo lo que tiene que coincidir entre la creación del índice (setup_index.py),
la ingesta (ingest.py) y la consulta (app/rag_system.py, evaluate.py) vive
acá, en un único lugar. En particular:

- El modelo de embeddings y su DIMENSIÓN: el índice de Pinecone se crea con
  `EMBEDDING_DIM` y los vectores se generan con ese mismo valor. Así se evita
  el error clásico de "Mismatch de Dimensiones" (subir vectores de 1536 a un
  índice de 768).
- El NAMESPACE: ingesta y consulta usan siempre el mismo, para no mezclar
  estos documentos con otros datos del índice.

Las claves y nombres se leen del archivo `.env` (nunca se hardcodean ni se
suben al repo: ver .gitignore y .env.example).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Rutas --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
GOLDEN_SET_PATH = DATA_DIR / "golden_set.json"

# --- Pinecone -----------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "python-docs-rag")
# Namespace = partición lógica del índice. Separa este corpus (docs de la
# librería estándar de Python) de cualquier otro dataset o entorno.
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "python-stdlib-es")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
# Coseno: los modelos de embeddings de Google y OpenAI están optimizados para
# similitud coseno (usar euclidiana sería otro de los errores comunes).
PINECONE_METRIC = "cosine"
# Batch Upsert: entre 100 y 200 vectores por request es el "punto dulce"
# entre velocidad y estabilidad.
UPSERT_BATCH_SIZE = 100

# --- Embeddings ----------------------------------------------------------
# Proveedores soportados:
#   google -> models/gemini-embedding-001 (tier gratuito de Google AI Studio)
#   openai -> text-embedding-3-small (el sugerido por la consigna)
# Ambos se configuran para producir vectores de 1536 dimensiones.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google").strip().lower()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_EMBEDDING_MODELS = {
    "google": "models/gemini-embedding-001",
    "openai": "text-embedding-3-small",
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or _EMBEDDING_MODELS.get(EMBEDDING_PROVIDER, "")
# Única fuente de verdad para la dimensión: setup_index.py crea el índice con
# este valor y app/embeddings.py le pide al modelo vectores de este tamaño.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# --- Chunking -----------------------------------------------------------
# Medido en tokens reales (tiktoken), no en caracteres. 600 tokens cae en el
# rango 500-800 sugerido: suficiente contexto semántico sin diluir el
# embedding. El overlap evita cortar una idea justo en el borde.
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 100
TOKENIZER_ENCODING = "cl100k_base"

# --- Recuperación -------------------------------------------------------
TOP_K = 5
# Cada retriever (BM25 y Pinecone) aporta más candidatos que el top-k final,
# para que la fusión RRF tenga material para reordenar antes de cortar.
CANDIDATES_PER_RETRIEVER = 10
# Pesos del EnsembleRetriever: [BM25 (léxico), Pinecone (semántico)].
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.5"))


def require_pinecone_key() -> str:
    """Falla rápido y con un mensaje claro si falta la API key de Pinecone."""
    if not PINECONE_API_KEY:
        raise RuntimeError(
            "Falta PINECONE_API_KEY. Copiá .env.example a .env y completá tu "
            "clave (gratis en https://app.pinecone.io)."
        )
    return PINECONE_API_KEY


def require_embedding_key() -> str:
    """Devuelve la API key del proveedor de embeddings configurado."""
    if EMBEDDING_PROVIDER == "google":
        key, name, url = GOOGLE_API_KEY, "GOOGLE_API_KEY", "https://aistudio.google.com/apikey"
    elif EMBEDDING_PROVIDER == "openai":
        key, name, url = OPENAI_API_KEY, "OPENAI_API_KEY", "https://platform.openai.com/api-keys"
    else:
        raise RuntimeError(
            f"EMBEDDING_PROVIDER='{EMBEDDING_PROVIDER}' no soportado. Usá 'google' u 'openai'."
        )
    if not key:
        raise RuntimeError(
            f"Falta {name} (EMBEDDING_PROVIDER={EMBEDDING_PROVIDER}). Completala en .env ({url})."
        )
    return key
