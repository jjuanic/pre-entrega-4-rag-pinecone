"""
Consulta interactiva al recuperador híbrido.

Uso:
    python query.py                                  # modo interactivo
    python query.py "¿qué es un TaskGroup?"          # una sola consulta
    python query.py --modo bm25 "ensure_ascii"       # comparar con un retriever solo
    python query.py --categoria concurrencia "..."   # hard filter por metadata
"""

from __future__ import annotations

import argparse
import sys

from app.rag_system import RAGSystem


def show(rag: RAGSystem, query: str, mode: str, metadata_filter: dict | None) -> None:
    results = rag.search(query, mode=mode, metadata_filter=metadata_filter)
    if not results:
        print("No se recuperó ningún fragmento.\n")
        return
    print(f"\nTop {len(results)} ({mode}):")
    for r in results:
        snippet = " ".join(r["contenido"].split())
        print(f"\n  {r['rank']}. [{r['fuente']} | p.{r['pagina']} | {r['categoria']}] {r['seccion']}")
        print(f"     {snippet[:220]}{'...' if len(snippet) > 220 else ''}")
    print("-" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta el RAGSystem híbrido.")
    parser.add_argument("consulta", nargs="*")
    parser.add_argument("--modo", choices=("hybrid", "bm25", "vector"), default="hybrid")
    parser.add_argument("--categoria", help="filtra por metadata category (hard filter)")
    args = parser.parse_args()

    metadata_filter = {"category": {"$eq": args.categoria}} if args.categoria else None
    rag = RAGSystem.from_pinecone()

    if args.consulta:
        show(rag, " ".join(args.consulta), args.modo, metadata_filter)
        return

    print("Recuperador híbrido (BM25 + Pinecone). Escribí 'salir' para terminar.\n")
    while True:
        query = input("Consulta: ").strip()
        if query.lower() in {"", "salir", "exit", "quit"}:
            break
        show(rag, query, args.modo, metadata_filter)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
