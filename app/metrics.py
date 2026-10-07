"""
Métricas de recuperación (funciones puras, sin dependencias de Pinecone).

Convenciones:
  - `retrieved`: lista ORDENADA de doc_id de los chunks recuperados
    (puede repetir un doc_id si se recuperaron varios chunks del mismo doc).
  - `relevant`: conjunto de doc_id que responden la pregunta (Golden Set).

Recall@k    = documentos relevantes distintos que aparecen en el top-k
              / total de documentos relevantes.
              Con un solo documento esperado vale 0 o 1 ("¿está o no está
              el documento correcto entre los k recuperados?").
Precision@k = chunks del top-k que pertenecen a un documento relevante / k.
              ("¿qué porcentaje de los k recuperados es realmente útil?")
"""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("Cada pregunta del Golden Set necesita al menos un documento relevante.")
    found = set(retrieved[:k]) & relevant
    return len(found) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    # Se divide por k (no por la cantidad recuperada): devolver menos
    # resultados no debería "premiar" la precisión.
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1 si al menos un documento relevante aparece en el top-k (Hit Rate)."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1/posición del primer resultado relevante (0 si no aparece). Su
    promedio sobre todas las preguntas es el MRR."""
    for position, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def precision_ceiling(relevant_chunk_count: int, k: int) -> float:
    """Máxima Precision@k alcanzable: si los documentos relevantes tienen
    en total 2 chunks, aunque el retriever sea perfecto la Precision@5 no
    puede superar 2/5 = 0.4."""
    return min(relevant_chunk_count, k) / k


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
