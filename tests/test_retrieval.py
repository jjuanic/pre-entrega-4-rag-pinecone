"""
Tests offline (no requieren API keys ni red): chunking + metadatos,
tokenizador BM25, filtros, fusión RRF del RAGSystem y métricas.

    python -m pytest -q
"""

from __future__ import annotations

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app import config
from app.documents import ChunkMetadata, chunk_documents, clean_text, load_documents, searchable_text
from app.metrics import f1, precision_at_k, precision_ceiling, recall_at_k, reciprocal_rank
from app.rag_system import RAGSystem
from app.retrievers import matches_filter, tokenize
from evaluate import evaluate, load_golden_set


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return [dict(c.metadata) for c in chunk_documents(load_documents())]


class FakeVectorRetriever(BaseRetriever):
    """Simula Pinecone devolviendo una lista fija de chunk_ids."""

    corpus: list[dict]
    ranking: list[str]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        by_id = {r["chunk_id"]: r for r in self.corpus}
        return [
            Document(page_content=by_id[i]["text"], metadata={k: v for k, v in by_id[i].items() if k != "text"})
            for i in self.ranking
        ]


# --- Ingesta: chunking y esquema de metadatos ---------------------------------


def test_chunks_respect_token_budget_and_schema(corpus):
    expected_keys = set(ChunkMetadata.__annotations__)
    for record in corpus:
        assert set(record) == expected_keys, "schema drift: todos los chunks deben tener el mismo esquema"
        assert record["n_tokens"] <= config.CHUNK_SIZE_TOKENS
        assert record["text"].strip()
        assert record["chunk_id"].startswith(f"{record['doc_id']}#")


def test_chunk_ids_are_unique_and_deterministic(corpus):
    ids = [r["chunk_id"] for r in corpus]
    assert len(ids) == len(set(ids))
    again = [c.id for c in chunk_documents(load_documents())]
    assert ids == again


def test_all_formats_and_pages(corpus):
    doc_types = {r["doc_type"] for r in corpus}
    assert {"markdown", "json"} <= doc_types
    faq_pages = {r["page"] for r in corpus if r["doc_id"] == "faq_errores_comunes"}
    assert len(faq_pages) > 1, "cada entrada del JSON es una página distinta"


def test_section_metadata_points_to_heading(corpus):
    chunk = next(r for r in corpus if r["text"].startswith("## Limitar la concurrencia"))
    assert chunk["section"] == "Limitar la concurrencia con Semaphore"
    assert "Semaphore" in searchable_text(chunk).splitlines()[0]


def test_clean_text_normalizes():
    assert clean_text("Hola mundo  \r\n\r\n\r\n\r\nchau\x00") == "Hola mundo\n\nchau"


# --- BM25 y filtros ---------------------------------------------------------


def test_tokenize_keeps_identifiers_and_drops_stopwords():
    tokens = tokenize("¿Qué hace el parámetro ensure_ascii de json.dumps?")
    assert "ensure_ascii" in tokens and "ensure" in tokens and "ascii" in tokens
    assert "json" in tokens and "dumps" in tokens and "parametro" in tokens
    assert "que" not in tokens and "de" not in tokens


def test_matches_filter_operators():
    meta = {"category": "concurrencia", "tags": ["async", "gil"]}
    assert matches_filter(meta, {"category": "concurrencia"})
    assert matches_filter(meta, {"tags": {"$in": ["gil", "x"]}})
    assert not matches_filter(meta, {"category": {"$ne": "concurrencia"}})
    assert not matches_filter(meta, {"tags": {"$nin": ["async"]}})


# --- RAGSystem: fusión RRF ---------------------------------------------------


def test_hybrid_fuses_lexical_and_semantic(corpus):
    # El "vectorial" falso pone primero un chunk de pathlib; BM25 encuentra
    # json por el término exacto. El híbrido debe traer ambos en el top-5.
    pathlib_chunk = next(r["chunk_id"] for r in corpus if r["doc_id"] == "pathlib")
    rag = RAGSystem(corpus=corpus, vector_retriever=FakeVectorRetriever(corpus=corpus, ranking=[pathlib_chunk]))
    docs = rag.retrieve("ensure_ascii json.dumps")
    doc_ids = [d.metadata["doc_id"] for d in docs]
    assert len(docs) == 5
    assert "json" in doc_ids[:2]
    assert "pathlib" in doc_ids
    assert [d.metadata["rank"] for d in docs] == [1, 2, 3, 4, 5]
    assert len({d.metadata["chunk_id"] for d in docs}) == 5, "sin duplicados"


def test_chunk_found_by_both_retrievers_ranks_first(corpus):
    target = next(r["chunk_id"] for r in corpus if "ensure_ascii" in r["text"])
    rag = RAGSystem(corpus=corpus, vector_retriever=FakeVectorRetriever(corpus=corpus, ranking=[target]))
    assert rag.retrieve("ensure_ascii")[0].metadata["chunk_id"] == target


def test_metadata_filter_applies_to_bm25(corpus):
    rag = RAGSystem(corpus=corpus, vector_retriever=FakeVectorRetriever(corpus=corpus, ranking=[]))
    docs = rag.retrieve("event loop corutinas", mode="bm25", metadata_filter={"category": "serializacion"})
    assert docs and all(d.metadata["category"] == "serializacion" for d in docs)


def test_bm25_mode_hits_golden_set(corpus):
    rag = RAGSystem(corpus=corpus, vector_retriever=FakeVectorRetriever(corpus=corpus, ranking=[]))
    report = evaluate(rag, load_golden_set(), k=5, mode="bm25")
    assert report["recall"] >= 0.8


# --- Métricas -----------------------------------------------------------------


def test_metrics():
    retrieved = ["a", "b", "a", "c", "d"]
    assert recall_at_k(retrieved, {"a"}, 5) == 1.0
    assert recall_at_k(retrieved, {"a", "z"}, 5) == 0.5
    assert recall_at_k(retrieved, {"d"}, 3) == 0.0
    assert precision_at_k(retrieved, {"a"}, 5) == pytest.approx(0.4)
    assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)  # se divide por k
    assert reciprocal_rank(retrieved, {"c"}, 5) == pytest.approx(0.25)
    assert precision_ceiling(3, 5) == pytest.approx(0.6)
    assert f1(0.5, 1.0) == pytest.approx(2 / 3)
