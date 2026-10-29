"""Tests for ChromaDB Schema & Metric Vector Retriever."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import chromadb
import pytest
import yaml
from chromadb.api.models.Collection import Collection
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


@pytest.fixture
def persistent_catalog(tmp_path):
    catalog = yaml.safe_load(Path(METRICS_PATH).read_text())
    path = tmp_path / "metrics.yaml"
    path.write_text(yaml.safe_dump(catalog))
    storage = tmp_path / "chroma"
    ret = SchemaMetricRetriever(storage_path=storage)
    ret.index_catalog(str(path), SAMPLE_SCHEMA)
    return ret, catalog, path, storage


def test_persistent_cache_reuses_unchanged_entries(persistent_catalog, monkeypatch):
    _, _, path, storage = persistent_catalog
    monkeypatch.setattr(Collection, "upsert", lambda *a, **kw: pytest.fail("unchanged entries re-embedded"))
    monkeypatch.setattr(Collection, "modify", lambda *a, **kw: pytest.fail("unchanged metadata rewritten"))
    reloaded = SchemaMetricRetriever(storage_path=storage)
    assert reloaded.index_catalog(str(path), SAMPLE_SCHEMA) == {
        "metrics_indexed": 5, "filters_indexed": 2, "tables_indexed": 4,
    }
    assert reloaded.metrics_collection.metadata["catalog_hash"]


def test_persistent_cache_survives_a_new_python_process(persistent_catalog):
    _, _, path, storage = persistent_catalog
    script = """
import sys
import json
from chromadb.api.models.Collection import Collection
from src.retriever import SchemaMetricRetriever
def no_embedding(*args, **kwargs):
    raise AssertionError('unchanged persistent entry was re-embedded')
Collection.upsert = no_embedding
retriever = SchemaMetricRetriever(storage_path=sys.argv[1])
schema = json.loads(sys.argv[3])
counts = retriever.index_catalog(sys.argv[2], schema)
assert counts == {'metrics_indexed': 5, 'filters_indexed': 2, 'tables_indexed': 4}
assert retriever.metrics_collection.get(ids=['metric_net_revenue'])['documents']
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(storage), str(path), json.dumps(SAMPLE_SCHEMA)],
        cwd=Path(__file__).resolve().parent.parent,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_catalog_change_only_reembeds_changed_metric(persistent_catalog, monkeypatch):
    ret, catalog, path, _ = persistent_catalog
    original_hash = ret.metrics_collection.metadata["catalog_hash"]
    catalog["metrics"][0]["formula"] = "SUM(l_extendedprice * 0.9)"
    path.write_text(yaml.safe_dump(catalog))
    upserts = []
    original_upsert = Collection.upsert

    def record_upsert(collection, **kwargs):
        upserts.append((collection.name, kwargs["ids"]))
        return original_upsert(collection, **kwargs)

    monkeypatch.setattr(Collection, "upsert", record_upsert)
    ret.index_catalog(str(path), SAMPLE_SCHEMA)
    assert upserts == [("metrics_catalog", ["metric_net_revenue"])]
    assert ret.metrics_collection.metadata["catalog_hash"] != original_hash
    assert ret.metrics_collection.get(ids=["metric_net_revenue"])["metadatas"][0]["formula"] == "SUM(l_extendedprice * 0.9)"


def test_schema_change_only_reembeds_changed_table(persistent_catalog, monkeypatch):
    ret, _, path, _ = persistent_catalog
    schema = {**SAMPLE_SCHEMA, "customer": [*SAMPLE_SCHEMA["customer"], {"name": "tier", "type": "VARCHAR"}]}
    upserts = []
    original_upsert = Collection.upsert

    def record_upsert(collection, **kwargs):
        upserts.append((collection.name, kwargs["ids"]))
        return original_upsert(collection, **kwargs)

    monkeypatch.setattr(Collection, "upsert", record_upsert)
    ret.index_catalog(str(path), schema)
    assert upserts == [("schema_catalog", ["table_customer"])]


def test_catalog_removals_prune_all_collections(persistent_catalog, monkeypatch):
    ret, catalog, path, _ = persistent_catalog
    catalog["metrics"].pop(0)
    catalog["filters"].pop(0)
    path.write_text(yaml.safe_dump(catalog))
    schema = {key: value for key, value in SAMPLE_SCHEMA.items() if key != "orders"}
    monkeypatch.setattr(Collection, "upsert", lambda *a, **kw: pytest.fail("unchanged entry re-embedded"))
    ret.index_catalog(str(path), schema)
    assert ret.metrics_collection.count() == 4
    assert ret.filters_collection.count() == 1
    assert ret.schema_collection.count() == 3
    assert ret.metrics_collection.get(ids=["metric_net_revenue"])["ids"] == []
    assert ret.filters_collection.get(ids=["filter_urgent_orders"])["ids"] == []
    assert ret.schema_collection.get(ids=["table_orders"])["ids"] == []


def test_empty_catalog_clears_persisted_entries(persistent_catalog):
    ret, _, path, _ = persistent_catalog
    path.write_text("metrics: []\nfilters: []\n")
    ret.index_catalog(str(path), {})
    assert ret.retrieve("What is the net revenue?") == ("", [])
    assert all(collection.count() == 0 for collection in (
        ret.metrics_collection, ret.filters_collection, ret.schema_collection,
    ))


def test_catalog_reordering_does_not_reembed(persistent_catalog, monkeypatch):
    ret, catalog, path, _ = persistent_catalog
    catalog["metrics"].reverse()
    catalog["filters"].reverse()
    path.write_text(yaml.safe_dump(catalog))
    monkeypatch.setattr(Collection, "upsert", lambda *a, **kw: pytest.fail("reordering re-embedded entries"))
    monkeypatch.setattr(Collection, "modify", lambda *a, **kw: pytest.fail("reordering changed hash"))
    ret.index_catalog(str(path), dict(reversed(list(SAMPLE_SCHEMA.items()))))


def test_failed_index_update_is_retried(persistent_catalog, monkeypatch):
    ret, catalog, path, _ = persistent_catalog
    original_hash = ret.metrics_collection.metadata["catalog_hash"]
    catalog["metrics"][0]["description"] = "Updated business definition"
    path.write_text(yaml.safe_dump(catalog))
    with monkeypatch.context() as patcher:
        def fail_update(*args, **kwargs):
            raise RuntimeError("interrupted indexing")
        patcher.setattr(Collection, "upsert", fail_update)
        with pytest.raises(RuntimeError, match="interrupted indexing"):
            ret.index_catalog(str(path), SAMPLE_SCHEMA)
    assert ret.metrics_collection.metadata["catalog_hash"] == original_hash
    ret.index_catalog(str(path), SAMPLE_SCHEMA)
    assert ret.metrics_collection.metadata["catalog_hash"] != original_hash


def test_missing_persisted_entry_is_restored(persistent_catalog):
    ret, _, path, _ = persistent_catalog
    ret.metrics_collection.delete(ids=["metric_tax_amount"])
    ret.index_catalog(str(path), SAMPLE_SCHEMA)
    assert ret.metrics_collection.count() == 5
    assert ret.metrics_collection.get(ids=["metric_tax_amount"])["ids"] == ["metric_tax_amount"]


def test_default_storage_is_relative_to_project_root(tmp_path, monkeypatch):
    from src import retriever as retriever_module

    monkeypatch.setattr(retriever_module, "app_settings", replace(
        retriever_module.app_settings, base_dir=tmp_path, chroma_storage_path=Path("vector_cache"),
    ))
    launch_directory = tmp_path / "another_directory"
    launch_directory.mkdir()
    monkeypatch.chdir(launch_directory)
    SchemaMetricRetriever()
    assert (tmp_path / "vector_cache").is_dir()
    assert not (launch_directory / "vector_cache").exists()
