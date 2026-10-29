"""Schema-Aware RAG and Business Metric Dictionary retriever using ChromaDB."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
import chromadb
from chromadb.config import Settings
from src.config import settings as app_settings


class SchemaMetricRetriever:
    """Persistent semantic retriever with content-hash-based incremental indexing."""

    def __init__(
        self,
        client: Optional[chromadb.ClientAPI] = None,
        storage_path: Optional[str | Path] = None,
    ):
        """Initialize persistent collections, or use a caller-supplied test/client.

        Args:
            client: Optional pre-configured ChromaDB ClientAPI instance.
            storage_path: Persistent directory, defaulting to CHROMA_STORAGE_PATH.
        """
        if client is not None:
            self.client = client
        else:
            target_path = Path(
                storage_path if storage_path is not None else app_settings.chroma_storage_path
            ).expanduser()
            if not target_path.is_absolute():
                target_path = app_settings.base_dir / target_path
            self.client = chromadb.PersistentClient(
                path=str(target_path.resolve()),
                settings=Settings(anonymized_telemetry=False)
            )

        self.metrics_collection = self.client.get_or_create_collection(
            name="metrics_catalog"
        )
        self.filters_collection = self.client.get_or_create_collection(
            name="filters_catalog"
        )
        self.schema_collection = self.client.get_or_create_collection(
            name="schema_catalog"
        )

    @staticmethod
    def _sync_collection(
        collection,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Embed changed entries only, prune removals, and persist a corpus hash.

        Mark the corpus current only after successful synchronization, so interrupted
        updates are retried on the next startup. Hashes include the index format version.
        """
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate catalog IDs in {collection.name}.")
        entries = []
        for item_id, document, metadata in zip(ids, documents, metadatas):
            payload = json.dumps(
                [1, document, metadata], sort_keys=True, ensure_ascii=False,
                separators=(",", ":"),
            )
            content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            entries.append((item_id, document, {**metadata, "content_hash": content_hash}))
        corpus_hash = hashlib.sha256(json.dumps(
            sorted((item_id, metadata["content_hash"]) for item_id, _, metadata in entries),
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

        stored = collection.get(include=["metadatas"])
        previous = dict(zip(stored["ids"], stored["metadatas"]))
        changed = [
            entry for entry in entries
            if (previous.get(entry[0]) or {}).get("content_hash") != entry[2]["content_hash"]
        ]
        for offset in range(0, len(changed), 100):
            batch = changed[offset:offset + 100]
            collection.upsert(
                ids=[entry[0] for entry in batch],
                documents=[entry[1] for entry in batch],
                metadatas=[entry[2] for entry in batch],
            )
        stale_ids = sorted(set(previous) - set(ids))
        for offset in range(0, len(stale_ids), 100):
            collection.delete(ids=stale_ids[offset:offset + 100])
        metadata = collection.metadata or {}
        if metadata.get("catalog_hash") != corpus_hash:
            collection.modify(metadata={**metadata, "catalog_hash": corpus_hash})

    def index_catalog(
        self,
        metrics_yaml_path: str,
        schema_info: Dict[str, List[Dict[str, str]]] | str,
    ) -> Dict[str, int]:
        """Index business metric catalog and schema metadata into ChromaDB.

        Args:
            metrics_yaml_path: Path to metrics.yaml file.
            schema_info: Schema dictionary or formatted DDL string.

        Returns:
            Dictionary with counts of indexed items.
        """
        metrics_path = Path(metrics_yaml_path).resolve()
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Metrics catalog not found at: {metrics_path}")

        with open(metrics_path, "r", encoding="utf-8") as f:
            catalog_data = yaml.safe_load(f) or {}

        # 1. Index Metrics (Aggregations)
        m_ids: List[str] = []
        m_docs: List[str] = []
        m_metadatas: List[Dict[str, Any]] = []

        for m in catalog_data.get("metrics", []):
            m_id = str(m.get("id", ""))
            name = str(m.get("name", m_id))
            formula = str(m.get("formula", ""))
            tables = ", ".join(m.get("tables", []))
            aliases = ", ".join(m.get("aliases", []))
            desc = str(m.get("description", ""))
            join_cond = str(m.get("join_condition", ""))

            doc_parts = [
                f"Business Metric: {name} ({m_id})",
                f"Type: Aggregation Formula",
                f"Formula: {formula}",
                f"Tables: {tables}",
                f"Description: {desc}",
                f"Aliases: {aliases}",
            ]
            if join_cond:
                doc_parts.append(f"Join Condition: {join_cond}")

            m_ids.append(f"metric_{m_id}")
            m_docs.append("\n".join(doc_parts))
            m_metadatas.append({
                "id": m_id,
                "name": name,
                "kind": "metric",
                "type": str(m.get("type", "aggregation")),
                "formula": formula,
                "tables": tables,
                "join_condition": join_cond,
                "description": desc,
            })

        self._sync_collection(self.metrics_collection, m_ids, m_docs, m_metadatas)

        # 2. Index Filters (Business Predicates)
        f_ids: List[str] = []
        f_docs: List[str] = []
        f_metadatas: List[Dict[str, Any]] = []

        for flt in catalog_data.get("filters", []):
            f_id = str(flt.get("id", ""))
            name = str(flt.get("name", f_id))
            condition = str(flt.get("condition", ""))
            tables = ", ".join(flt.get("tables", []))
            aliases = ", ".join(flt.get("aliases", []))
            desc = str(flt.get("description", ""))
            join_cond = str(flt.get("join_condition", ""))

            doc_parts = [
                f"Business Filter Rule: {name} ({f_id})",
                f"Type: Filter Predicate",
                f"Condition: {condition}",
                f"Tables: {tables}",
                f"Description: {desc}",
                f"Aliases: {aliases}",
            ]
            if join_cond:
                doc_parts.append(f"Join Condition: {join_cond}")

            f_ids.append(f"filter_{f_id}")
            f_docs.append("\n".join(doc_parts))
            f_metadatas.append({
                "id": f_id,
                "name": name,
                "kind": "filter",
                "type": str(flt.get("type", "filter")),
                "condition": condition,
                "tables": tables,
                "join_condition": join_cond,
                "description": desc,
            })

        self._sync_collection(self.filters_collection, f_ids, f_docs, f_metadatas)

        # 3. Index Schema Tables & Columns
        s_ids: List[str] = []
        s_docs: List[str] = []
        s_metadatas: List[Dict[str, Any]] = []

        if isinstance(schema_info, dict):
            for tbl_name, cols in schema_info.items():
                col_defs = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
                doc = f"Table: {tbl_name}\nColumns: {col_defs}"
                s_ids.append(f"table_{tbl_name}")
                s_docs.append(doc)
                s_metadatas.append({
                    "table": tbl_name,
                    "col_count": len(cols),
                })
        elif isinstance(schema_info, str):
            for line in schema_info.splitlines():
                line_str = line.strip()
                if line_str.startswith("CREATE TABLE"):
                    parts = line_str.split("(", 1)
                    tbl_name = parts[0].replace("CREATE TABLE", "").strip()
                    s_ids.append(f"table_{tbl_name}")
                    s_docs.append(line_str)
                    s_metadatas.append({"table": tbl_name, "col_count": 0})

        self._sync_collection(self.schema_collection, s_ids, s_docs, s_metadatas)

        return {
            "metrics_indexed": len(m_ids),
            "filters_indexed": len(f_ids),
            "tables_indexed": len(s_ids),
        }

    def retrieve(
        self,
        user_question: str,
        top_k_metrics: int = 2,
        top_k_filters: int = 2,
        top_k_tables: int = 3,
        distance_threshold: float = 1.35,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Perform semantic search against indexed metrics, filters, and tables.

        Args:
            user_question: The natural language user query.
            top_k_metrics: Maximum number of metric aggregation candidates.
            top_k_filters: Maximum number of filter predicate candidates.
            top_k_tables: Maximum number of table candidates.
            distance_threshold: Maximum distance to consider relevant.

        Returns:
            Tuple of (formatted_markdown_context_str, list_of_retrieved_metadata).
        """
        if not user_question or not user_question.strip():
            return "", []

        retrieved_metadata: List[Dict[str, Any]] = []
        metric_lines: List[str] = []
        filter_lines: List[str] = []
        table_lines: List[str] = []

        # 1. Query Metrics (Aggregations)
        total_metrics = self.metrics_collection.count()
        if total_metrics > 0:
            k_m = min(top_k_metrics, total_metrics)
            m_res = self.metrics_collection.query(
                query_texts=[user_question],
                n_results=k_m,
            )
            if m_res and m_res["ids"] and m_res["ids"][0]:
                for idx, item_id in enumerate(m_res["ids"][0]):
                    dist = m_res["distances"][0][idx] if m_res.get("distances") else 0.0
                    meta = m_res["metadatas"][0][idx] if m_res.get("metadatas") else {}
                    if dist <= distance_threshold:
                        retrieved_metadata.append(meta)
                        line = (
                            f"- **{meta.get('name')}** (`{meta.get('id')}`): "
                            f"Formula: `{meta.get('formula')}` | Tables: `{meta.get('tables')}`"
                        )
                        if meta.get("join_condition"):
                            line += f" | Join: `{meta.get('join_condition')}`"
                        metric_lines.append(line)

        # 2. Query Filters (Business Predicates)
        total_filters = self.filters_collection.count()
        if total_filters > 0:
            k_f = min(top_k_filters, total_filters)
            f_res = self.filters_collection.query(
                query_texts=[user_question],
                n_results=k_f,
            )
            if f_res and f_res["ids"] and f_res["ids"][0]:
                for idx, item_id in enumerate(f_res["ids"][0]):
                    dist = f_res["distances"][0][idx] if f_res.get("distances") else 0.0
                    meta = f_res["metadatas"][0][idx] if f_res.get("metadatas") else {}
                    if dist <= distance_threshold:
                        retrieved_metadata.append(meta)
                        line = (
                            f"- **{meta.get('name')}** (`{meta.get('id')}`): "
                            f"Filter Condition: `{meta.get('condition')}` | Tables: `{meta.get('tables')}`"
                        )
                        if meta.get("join_condition"):
                            line += f" | Join: `{meta.get('join_condition')}`"
                        filter_lines.append(line)

        # 3. Query Schema Tables
        total_tables = self.schema_collection.count()
        if total_tables > 0:
            k_t = min(top_k_tables, total_tables)
            t_res = self.schema_collection.query(
                query_texts=[user_question],
                n_results=k_t,
            )
            if t_res and t_res["ids"] and t_res["ids"][0]:
                for idx, tbl_id in enumerate(t_res["ids"][0]):
                    dist = t_res["distances"][0][idx] if t_res.get("distances") else 0.0
                    meta = t_res["metadatas"][0][idx] if t_res.get("metadatas") else {}
                    if dist <= distance_threshold:
                        tbl_name = meta.get("table", "")
                        if tbl_name:
                            table_lines.append(f"- Table `{tbl_name}`")

        # 4. Format Combined Context
        sections: List[str] = []
        if metric_lines:
            sections.append("### Business Metrics (Formulas):\n" + "\n".join(metric_lines))
        if filter_lines:
            sections.append("### Business Rules & Filters (Predicates):\n" + "\n".join(filter_lines))
        if table_lines:
            sections.append("### Semantically Relevant Tables:\n" + "\n".join(table_lines))

        rag_context = "\n\n".join(sections)
        return rag_context, retrieved_metadata
