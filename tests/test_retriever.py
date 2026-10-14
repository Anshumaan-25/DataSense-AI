"""Tests for ChromaDB Schema & Metric Vector Retriever."""

import pytest
import chromadb
from chromadb.config import Settings
from src.retriever import SchemaMetricRetriever

METRICS_PATH = "data/metrics.yaml"

SAMPLE_SCHEMA = {
    "customer": [
        {"name": "c_custkey", "type": "BIGINT"},
        {"name": "c_name", "type": "VARCHAR"},
        {"name": "c_nationkey", "type": "INTEGER"},
    ],
    "lineitem": [
        {"name": "l_orderkey", "type": "BIGINT"},
        {"name": "l_extendedprice", "type": "DECIMAL(15,2)"},
        {"name": "l_discount", "type": "DECIMAL(15,2)"},
        {"name": "l_tax", "type": "DECIMAL(15,2)"},
    ],
    "nation": [
        {"name": "n_nationkey", "type": "INTEGER"},
        {"name": "n_name", "type": "VARCHAR"},
    ],
    "orders": [
        {"name": "o_orderkey", "type": "BIGINT"},
        {"name": "o_orderpriority", "type": "VARCHAR"},
    ],
}


@pytest.fixture(scope="module")
def retriever():
    """Create a persistent test instance of SchemaMetricRetriever with in-memory ChromaDB."""
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    ret = SchemaMetricRetriever(client=client)
    ret.index_catalog(METRICS_PATH, SAMPLE_SCHEMA)
    return ret


def test_index_catalog_counts(retriever):
    """Verify metrics, filters, and tables are correctly indexed into ChromaDB collections."""
    assert retriever.metrics_collection.count() == 5
    assert retriever.filters_collection.count() == 2
    assert retriever.schema_collection.count() == 4


def test_retrieve_net_revenue_germany(retriever):
    """Verify question about net revenue for Germany retrieves formula and filter."""
    question = "What is the net revenue for customers in Germany?"
    context, metadata = retriever.retrieve(question, top_k_metrics=2, top_k_filters=2, top_k_tables=3)

    assert "SUM(l_extendedprice * (1 - l_discount))" in context
    assert "n_name = 'GERMANY'" in context
    assert "customer.c_nationkey = nation.n_nationkey" in context

    retrieved_ids = {m.get("id") for m in metadata}
    assert "net_revenue" in retrieved_ids
    assert "german_customers" in retrieved_ids


def test_retrieve_item_margin_urgent_orders(retriever):
    """Verify question about item margin and urgent orders retrieves composite join and priority condition."""
    question = "Calculate the item margin for urgent orders."
    context, metadata = retriever.retrieve(question, top_k_metrics=2, top_k_filters=2, top_k_tables=3)

    assert "ps_supplycost * l_quantity" in context
    assert "l_partkey = partsupp.ps_partkey AND lineitem.l_suppkey = partsupp.ps_suppkey" in context
    assert "1-URGENT" in context

    retrieved_ids = {m.get("id") for m in metadata}
    assert "item_margin" in retrieved_ids
    assert "urgent_orders" in retrieved_ids


def test_retrieve_empty_question(retriever):
    """Verify empty query returns empty context and empty metadata."""
    context, metadata = retriever.retrieve("   ")
    assert context == ""
    assert metadata == []


def test_retrieve_irrelevant_threshold(retriever):
    """Verify strict distance threshold suppresses irrelevant metrics for off-topic query."""
    # With a strict threshold (e.g. 0.5), completely off-topic questions should return empty metrics
    context, metadata = retriever.retrieve(
        "Who was the 16th president of the United States?",
        distance_threshold=0.5,
    )
    assert metadata == []
    assert context == ""
