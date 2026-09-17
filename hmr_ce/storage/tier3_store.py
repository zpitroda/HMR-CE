"""
Tier 3: Topological Macro-Centroid Map (Coarse Resolution d=64)
A compact index of conversational themes, tasks, or temporal phases.
Tracks unit-normalized coarse centroids, member counts, and bounding turn ranges.
Enforces Surprise Isolation to prevent semantic smearing.
"""

import sqlite3
import json
from pathlib import Path
from threading import Lock
from typing import List, Optional, Union, Tuple
import numpy as np

from hmr_ce.schemas import Tier3MacroCentroid


class Tier3Store:
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
                    CREATE TABLE IF NOT EXISTS tier3_macro_centroids (
                        cluster_id TEXT PRIMARY KEY,
                        centroid_coarse BLOB NOT NULL,
                        member_count INTEGER NOT NULL DEFAULT 1,
                        min_turn_id INTEGER NOT NULL,
                        max_turn_id INTEGER NOT NULL
                    );
                """)
                conn.commit()

    def create_cluster(self, cluster_id: str, initial_coarse: Union[np.ndarray, List[float]], turn_id: int) -> Tier3MacroCentroid:
        """Instantiate a new thematic cluster with a unit-normalized coarse embedding."""
        initial_coarse = np.asarray(initial_coarse, dtype=np.float32)
        norm = np.linalg.norm(initial_coarse)
        if norm > 1e-8:
            unit_coarse = (initial_coarse / norm).astype(np.float32)
        else:
            unit_coarse = initial_coarse.astype(np.float32)

        record = Tier3MacroCentroid(
            cluster_id=cluster_id,
            centroid_coarse=unit_coarse.tolist(),
            member_count=1,
            bounding_turn_range=(turn_id, turn_id)
        )

        blob = unit_coarse.tobytes()
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tier3_macro_centroids (
                        cluster_id, centroid_coarse, member_count, min_turn_id, max_turn_id
                    ) VALUES (?, ?, ?, ?, ?)
                """, (cluster_id, blob, 1, turn_id, turn_id))
                conn.commit()

        return record

    def aggregate_node(self, cluster_id: str, node_coarse: Union[np.ndarray, List[float]], turn_id: int) -> Optional[Tier3MacroCentroid]:
        """
        Aggregate a new constituent node into running centroid:
        C_macro = (C_macro * n + e_coarse) / (n + 1)
        C_macro = C_macro / ||C_macro||_2
        """
        node_coarse = np.asarray(node_coarse, dtype=np.float32)
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tier3_macro_centroids WHERE cluster_id = ?", (cluster_id,))
                row = cursor.fetchone()
                if not row:
                    return None

                old_blob = row["centroid_coarse"]
                old_centroid = np.frombuffer(old_blob, dtype=np.float32)
                old_count = row["member_count"]
                min_t = min(row["min_turn_id"], turn_id)
                max_t = max(row["max_turn_id"], turn_id)

                # Running average update
                new_centroid = (old_centroid * old_count + node_coarse) / (old_count + 1)
                norm = np.linalg.norm(new_centroid)
                if norm > 1e-8:
                    new_centroid = new_centroid / norm
                new_centroid = new_centroid.astype(np.float32)
                new_count = old_count + 1

                new_blob = new_centroid.tobytes()
                cursor.execute("""
                    UPDATE tier3_macro_centroids
                    SET centroid_coarse = ?, member_count = ?, min_turn_id = ?, max_turn_id = ?
                    WHERE cluster_id = ?
                """, (new_blob, new_count, min_t, max_t, cluster_id))
                conn.commit()

                return Tier3MacroCentroid(
                    cluster_id=cluster_id,
                    centroid_coarse=new_centroid.tolist(),
                    member_count=new_count,
                    bounding_turn_range=(min_t, max_t)
                )

    def get_cluster(self, cluster_id: str) -> Optional[Tier3MacroCentroid]:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tier3_macro_centroids WHERE cluster_id = ?", (cluster_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_centroid(row)

    def get_all_clusters(self) -> List[Tier3MacroCentroid]:
        """Fetch all active Tier 3 clusters for Phase 1 Macro-Sweeps."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tier3_macro_centroids ORDER BY min_turn_id ASC")
                return [self._row_to_centroid(r) for r in cursor.fetchall()]

    def clear(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM tier3_macro_centroids")
                conn.commit()

    def close(self) -> None:
        """Close SQLite database connection."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    @staticmethod
    def _row_to_centroid(row: sqlite3.Row) -> Tier3MacroCentroid:
        blob = row["centroid_coarse"]
        centroid_coarse = np.frombuffer(blob, dtype=np.float32).tolist()
        return Tier3MacroCentroid(
            cluster_id=row["cluster_id"],
            centroid_coarse=centroid_coarse,
            member_count=row["member_count"],
            bounding_turn_range=(row["min_turn_id"], row["max_turn_id"])
        )
