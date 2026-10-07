"""
Setup de Pinecone: verifica si el índice existe y, si no, lo crea en modo
Serverless con la dimensión y métrica del modelo de embeddings configurado.

Es idempotente: correrlo varias veces no crea nada nuevo. Si el índice ya
existe pero con otra dimensión o métrica, falla con un mensaje claro (en vez
de dejar que la ingesta explote más adelante).

Uso:
    python setup_index.py
"""

from __future__ import annotations

from app import config
from app.pinecone_store import ensure_index, get_index


def main() -> None:
    print(f"Índice:     {config.INDEX_NAME}")
    print(f"Spec:       Serverless ({config.PINECONE_CLOUD} / {config.PINECONE_REGION})")
    print(f"Dimensión:  {config.EMBEDDING_DIM}  ({config.EMBEDDING_PROVIDER}: {config.EMBEDDING_MODEL})")
    print(f"Métrica:    {config.PINECONE_METRIC}")
    print(f"Namespace:  {config.NAMESPACE}\n")

    created = ensure_index()
    if created:
        print(f"Índice '{config.INDEX_NAME}' creado y listo.")
    else:
        print(f"El índice '{config.INDEX_NAME}' ya existía (dimensión y métrica verificadas): no se recrea.")

    stats = get_index().describe_index_stats()
    print(f"\nVectores totales: {stats.total_vector_count}")
    for name, summary in (stats.namespaces or {}).items():
        print(f"  - namespace '{name}': {summary.vector_count} vectores")


if __name__ == "__main__":
    main()
