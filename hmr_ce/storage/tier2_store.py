"""
Tier 2: Episodic Memory Store (Fine Resolution)
Vector-indexed record representing an atomic assertion, subtask, or conversational episode.
Tracks surprise salience, momentum inheritance, directed supersession graph edges,
and cluster memberships.
"""

import sqlite3
import json
from pathlib import Path
from threading import Lock
from typing import List, Optional, Union, Dict, Any
import numpy as np

from hmr_ce.schemas import Tier2EpisodicNode, ActiveStatus


class Tier2Store:
    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        self.db_path = str(db_path)
        self._lock = Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tier2_episodic_store (
                        node_id TEXT PRIMARY KEY,
                        turn_references TEXT NOT NULL,
                        embedding_fine BLOB,
                        embedding_coarse BLOB,
                        surprise_salience REAL NOT NULL,
                        momentum_inherited INTEGER NOT NULL DEFAULT 0,
                        centroid_cluster_id TEXT NOT NULL,
                        superseded_by_node TEXT,
                        status TEXT NOT NULL DEFAULT 'ACTIVE',
                        metadata TEXT NOT NULL DEFAULT '{}'
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tier2_cluster ON tier2_episodic_store (centroid_cluster_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tier2_status ON tier2_episodic_store (status);")
                conn.commit()

    def insert_node(self, node: Tier2EpisodicNode) -> None:
        """Insert or update a Tier 2 episodic memory node."""
        fine_blob = np.array(node.embedding_fine, dtype=np.float32).tobytes() if node.embedding_fine else None
        coarse_blob = np.array(node.embedding_coarse, dtype=np.float32).tobytes() if node.embedding_coarse else None

        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tier2_episodic_store (
                        node_id, turn_references, embedding_fine, embedding_coarse,
                        surprise_salience, momentum_inherited, centroid_cluster_id,
                        superseded_by_node, status, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    node.node_id,
                    json.dumps(node.turn_references),
                    fine_blob,
                    coarse_blob,
                    float(node.surprise_salience),
                    1 if node.momentum_inherited else 0,
                    node.centroid_cluster_id,
                    node.superseded_by_node,
                    node.status.value,
                    json.dumps(node.metadata)
                ))
                conn.commit()

    def get_node(self, node_id: str) -> Optional[Tier2EpisodicNode]:
        """Fetch an episodic node by node_id."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tier2_episodic_store WHERE node_id = ?", (node_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_node(row)

    def mark_superseded(self, target_node_id: str, overriding_node_id: str) -> bool:
        """
        Directed Belief Revision transition:
        Update target node status from ACTIVE to SUPERSEDED,
        insert directed link superseded_by_node, and suppress standard salience.
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE tier2_episodic_store
                    SET status = 'SUPERSEDED', superseded_by_node = ?
                    WHERE node_id = ?
                """, (overriding_node_id, target_node_id))
                conn.commit()
                return cursor.rowcount > 0

    def get_nodes_by_cluster(self, cluster_id: str, only_active: bool = True) -> List[Tier2EpisodicNode]:
        """Fetch nodes belonging to a specific cluster, optionally filtering by ACTIVE status."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if only_active:
                    cursor.execute("""
                        SELECT * FROM tier2_episodic_store
                        WHERE centroid_cluster_id = ? AND status = 'ACTIVE'
                    """, (cluster_id,))
                else:
                    cursor.execute("""
                        SELECT * FROM tier2_episodic_store
                        WHERE centroid_cluster_id = ?
                    """, (cluster_id,))
                return [self._row_to_node(r) for r in cursor.fetchall()]

    def get_active_nodes_in_clusters(self, cluster_ids: List[str]) -> List[Tier2EpisodicNode]:
        """Gather candidate active Tier 2 nodes belonging to Top-K clusters."""
        if not cluster_ids:
            return []
        placeholders = ",".join("?" for _ in cluster_ids)
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT * FROM tier2_episodic_store
                    WHERE centroid_cluster_id IN ({placeholders}) AND status = 'ACTIVE'
                """, cluster_ids)
                return [self._row_to_node(r) for r in cursor.fetchall()]

    def get_all_nodes(self, only_active: bool = False) -> List[Tier2EpisodicNode]:
        """Fetch all episodic nodes (useful for retrospective search and graph visualization)."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if only_active:
                    cursor.execute("SELECT * FROM tier2_episodic_store WHERE status = 'ACTIVE'")
                else:
                    cursor.execute("SELECT * FROM tier2_episodic_store")
                return [self._row_to_node(r) for r in cursor.fetchall()]

    def clear(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM tier2_episodic_store")
                conn.commit()

    def close(self) -> None:
        """Close SQLite database connection."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> Tier2EpisodicNode:
        turn_refs = json.loads(row["turn_references"])
        metadata = json.loads(row["metadata"])
        
        fine_blob = row["embedding_fine"]
        embedding_fine = np.frombuffer(fine_blob, dtype=np.float32).tolist() if fine_blob else None

        coarse_blob = row["embedding_coarse"]
        embedding_coarse = np.frombuffer(coarse_blob, dtype=np.float32).tolist() if coarse_blob else None

        return Tier2EpisodicNode(
            node_id=row["node_id"],
            turn_references=turn_refs,
            embedding_fine=embedding_fine,
            embedding_coarse=embedding_coarse,
            surprise_salience=float(row["surprise_salience"]),
            momentum_inherited=bool(row["momentum_inherited"]),
            centroid_cluster_id=row["centroid_cluster_id"],
            superseded_by_node=row["superseded_by_node"],
            status=ActiveStatus(row["status"]),
            metadata=metadata
        )
