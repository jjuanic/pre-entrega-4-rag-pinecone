"""
Evaluación cuantitativa del recuperador con un Golden Set.

Para cada pregunta de data/golden_set.json (cuyo documento fuente se conoce
de antemano) se recuperan los top-k chunks y se calcula:

  - Recall@k:    ¿está el documento correcto entre los k recuperados?
  - Precision@k: ¿qué porcentaje de los k recuperados es realmente útil
                 (pertenece a un documento relevante)?
  - Extras: Hit Rate, MRR y F1.

Por defecto evalúa el sistema HÍBRIDO y además lo compara contra BM25 solo y
Pinecone solo, para mostrar qué aporta la combinación.

Uso:
    python evaluate.py                    # híbrido (detalle) + comparación de los 3 modos
    python evaluate.py --modo hybrid      # solo el híbrido
    python evaluate.py --k 3              # otro k
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from app import config
from app.metrics import f1, hit_at_k, mean, precision_at_k, precision_ceiling, recall_at_k, reciprocal_rank
from app.rag_system import RAGSystem

MODES = ("hybrid", "bm25", "vector")
MODE_LABELS = {"hybrid": "Híbrido (BM25 + Pinecone)", "bm25": "Solo BM25", "vector": "Solo Pinecone"}


def load_golden_set(path: Path = config.GOLDEN_SET_PATH) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    for case in cases:
        expected = case["documento_id_esperado"]
        case["relevantes"] = {expected} if isinstance(expected, str) else set(expected)
    return cases


def evaluate(rag: RAGSystem, golden_set: list[dict], k: int, mode: str) -> dict:
    chunks_per_doc = Counter(record["doc_id"] for record in rag.corpus)
    rows = []
    for case in golden_set:
        docs = rag.retrieve(case["pregunta"], k=k, mode=mode)
        retrieved = [d.metadata["doc_id"] for d in docs]
        relevant = case["relevantes"]
        precision = precision_at_k(retrieved, relevant, k)
        recall = recall_at_k(retrieved, relevant, k)
        rows.append(
            {
                "pregunta": case["pregunta"],
                "tipo": case.get("tipo", "-"),
                "esperado": sorted(relevant),
                "recuperados": retrieved,
                "recall": recall,
                "precision": precision,
                "techo_precision": precision_ceiling(sum(chunks_per_doc[d] for d in relevant), k),
                "hit": hit_at_k(retrieved, relevant, k),
                "rr": reciprocal_rank(retrieved, relevant, k),
            }
        )
    recall_avg = mean([r["recall"] for r in rows])
    precision_avg = mean([r["precision"] for r in rows])
    return {
        "modo": mode,
        "k": k,
        "detalle": rows,
        "recall": recall_avg,
        "precision": precision_avg,
        "techo_precision": mean([r["techo_precision"] for r in rows]),
        "hit_rate": mean([r["hit"] for r in rows]),
        "mrr": mean([r["rr"] for r in rows]),
        "f1": f1(precision_avg, recall_avg),
    }


def print_detail(report: dict) -> None:
    k = report["k"]
    print("=" * 88)
    print(f"DETALLE — {MODE_LABELS[report['modo']]} — top-{k}")
    print("=" * 88)
    for i, row in enumerate(report["detalle"], start=1):
        status = "OK " if row["recall"] == 1.0 else ("~  " if row["recall"] > 0 else "XX ")
        print(f"[{status}] {i:>2}. ({row['tipo']}) {row['pregunta']}")
        print(f"         Esperado:    {', '.join(row['esperado'])}")
        print(f"         Recuperados: {row['recuperados']}")
        print(
            f"         Recall@{k}: {row['recall']:.0%} | Precision@{k}: {row['precision']:.0%} "
            f"(techo {row['techo_precision']:.0%}) | RR: {row['rr']:.2f}\n"
        )


def print_summary(reports: list[dict]) -> None:
    k = reports[0]["k"]
    n = len(reports[0]["detalle"])
    print("=" * 88)
    print(f"RESUMEN — {n} preguntas del Golden Set, top-{k}, índice '{config.INDEX_NAME}', "
          f"namespace '{config.NAMESPACE}'")
    print("=" * 88)
    header = f"{'Modo':<28}{'Recall@' + str(k):>10}{'Precision@' + str(k):>14}{'Hit Rate':>10}{'MRR':>8}{'F1':>8}"
    print(header)
    print("-" * len(header))
    for r in reports:
        print(
            f"{MODE_LABELS[r['modo']]:<28}{r['recall']:>10.1%}{r['precision']:>14.1%}"
            f"{r['hit_rate']:>10.1%}{r['mrr']:>8.3f}{r['f1']:>8.3f}"
        )
    print("-" * len(header))
    print(
        f"Techo teórico de Precision@{k} con este Golden Set: {reports[0]['techo_precision']:.1%} "
        "(los documentos relevantes no siempre tienen 5 chunks)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa Recall@k y Precision@k sobre el Golden Set.")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--modo", choices=(*MODES, "todos"), default="todos")
    parser.add_argument("--golden", type=Path, default=config.GOLDEN_SET_PATH)
    args = parser.parse_args()

    golden_set = load_golden_set(args.golden)
    print(f"Cargando RAGSystem desde Pinecone (índice '{config.INDEX_NAME}')...")
    rag = RAGSystem.from_pinecone(k=args.k)
    print(f"Corpus: {len(rag.corpus)} chunks | pesos RRF [BM25, vector] = {rag.weights}\n")

    modes = MODES if args.modo == "todos" else (args.modo,)
    reports = [evaluate(rag, golden_set, args.k, mode) for mode in modes]
    print_detail(reports[0])
    print_summary(reports)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
