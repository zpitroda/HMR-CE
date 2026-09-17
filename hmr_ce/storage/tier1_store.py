"""
Tier 1: Ground-Truth Pointer Store (Immutable Disk Storage)
An append-only relational SQLite store indexed monotonically by turn_id.
Decouples semantic index traversal from literal text verification,
guaranteeing 100% factual and verbatim fidelity when hydrated.
"""

import sqlite3
import json
from pathlib import Path
from threading import Lock
from typing import List, Optional, Union
from datetime import datetime

from hmr_ce.schemas import Tier1Record, SpeakerRole, ActiveStatus


class Tier1Store:
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
                    CREATE TABLE IF NOT EXISTS tier1_pointer_store (
                        turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        speaker_role TEXT NOT NULL,
                        raw_text TEXT NOT NULL,
                        token_count INTEGER NOT NULL DEFAULT 0,
                        active_status TEXT NOT NULL DEFAULT 'ACTIVE'
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tier1_session ON tier1_pointer_store (session_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tier1_status ON tier1_pointer_store (active_status);")
                conn.commit()

    def insert_turn(
        self,
        session_id: str,
        speaker_role: SpeakerRole,
        raw_text: str,
        token_count: int = 0,
        active_status: ActiveStatus = ActiveStatus.ACTIVE,
        timestamp: Optional[str] = None
    ) -> Tier1Record:
        """Append a new verbatim turn payload. Returns the created Tier1Record with monotonic turn_id."""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO tier1_pointer_store (session_id, timestamp, speaker_role, raw_text, token_count, active_status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, timestamp, speaker_role.value, raw_text, token_count, active_status.value))
                turn_id = cursor.lastrowid
                conn.commit()

        return Tier1Record(
            turn_id=turn_id,
            session_id=session_id,
            timestamp=timestamp,
            speaker_role=speaker_role,
            raw_text=raw_text,
            token_count=token_count,
            active_status=active_status
        )

    def get_turn(self, turn_id: int) -> Optional[Tier1Record]:
        """Fetch verbatim payload by monotonic turn_id."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tier1_pointer_store WHERE turn_id = ?", (turn_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_record(row)

    def get_turns_batch(self, turn_ids: List[int]) -> List[Tier1Record]:
        """Batch fetch verbatim payloads preserving turn_id ordering."""
        if not turn_ids:
            return []
        placeholders = ",".join("?" for _ in turn_ids)
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM tier1_pointer_store WHERE turn_id IN ({placeholders})", turn_ids)
                rows = cursor.fetchall()
                record_map = {row["turn_id"]: self._row_to_record(row) for row in rows}
                return [record_map[tid] for tid in turn_ids if tid in record_map]

    def update_status(self, turn_id: int, status: ActiveStatus) -> bool:
        """Update active_status of a turn (e.g. from ACTIVE to SUPERSEDED)."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE tier1_pointer_store SET active_status = ? WHERE turn_id = ?", (status.value, turn_id))
                conn.commit()
                return cursor.rowcount > 0

    def get_all_turns(self, session_id: Optional[str] = None) -> List[Tier1Record]:
        """Retrieve all turns, optionally filtered by session."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if session_id:
                    cursor.execute("SELECT * FROM tier1_pointer_store WHERE session_id = ? ORDER BY turn_id ASC", (session_id,))
                else:
                    cursor.execute("SELECT * FROM tier1_pointer_store ORDER BY turn_id ASC")
                return [self._row_to_record(r) for r in cursor.fetchall()]

    def get_max_turn_id(self) -> int:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(turn_id) as max_id FROM tier1_pointer_store")
                row = cursor.fetchone()
                return row["max_id"] if row and row["max_id"] is not None else 0

    def clear(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM tier1_pointer_store")
                try:
                    conn.execute("DELETE FROM sqlite_sequence WHERE name = 'tier1_pointer_store'")
                except sqlite3.OperationalError:
                    pass
                conn.commit()

    def close(self) -> None:
        """Close SQLite database connection."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> Tier1Record:
        return Tier1Record(
            turn_id=row["turn_id"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            speaker_role=SpeakerRole(row["speaker_role"]),
            raw_text=row["raw_text"],
            token_count=row["token_count"],
            active_status=ActiveStatus(row["active_status"])
        )
