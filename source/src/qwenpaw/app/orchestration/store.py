"""SQLite persistence for orchestration tasks and idempotency keys."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from ...constant import WORKING_DIR


class TaskStore:
    """Persist complete task snapshots with optimistic revisions."""

    def __init__(self, workspace_dir: Path | None = None) -> None:
        root = Path(workspace_dir or (WORKING_DIR / "working"))
        self.path = root / "orchestration" / "tasks.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS orchestration_tasks (
                    id TEXT PRIMARY KEY,
                    request_id TEXT UNIQUE,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._invalidate_legacy_completions(connection)

    @staticmethod
    def _invalidate_legacy_completions(connection: sqlite3.Connection) -> None:
        """Do not preserve success claims made before completion contracts existed."""
        rows = connection.execute(
            "SELECT id, revision, payload FROM orchestration_tasks WHERE status = ?",
            ("completed",),
        ).fetchall()
        for row in rows:
            task = json.loads(row["payload"])
            if (
                task.get("contract_version")
                and task.get("goal_contract")
                and task.get("terminal_predicates")
            ):
                continue
            now = time.time()
            task["status"] = "blocked"
            task["revision"] = int(row["revision"]) + 1
            task["updated_at"] = now
            task["blocked_reason"] = (
                "legacy completion invalidated: no immutable goal contract or "
                "domain-backed terminal evidence"
            )
            task["legacy_completion_invalidated_at"] = now
            connection.execute(
                """
                UPDATE orchestration_tasks
                SET status = ?, revision = ?, updated_at = ?, payload = ?
                WHERE id = ? AND revision = ?
                """,
                (
                    task["status"],
                    task["revision"],
                    task["updated_at"],
                    json.dumps(task, ensure_ascii=False),
                    row["id"],
                    row["revision"],
                ),
            )

    def create(self, task: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Create a task, or return the existing task for the same request id."""
        with self._lock, self._connect() as connection:
            request_id = task.get("request_id")
            if request_id:
                row = connection.execute(
                    "SELECT payload FROM orchestration_tasks WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                if row:
                    return json.loads(row["payload"]), False
            connection.execute(
                """
                INSERT INTO orchestration_tasks
                    (id, request_id, status, revision, updated_at, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task["id"],
                    request_id,
                    task["status"],
                    task["revision"],
                    task["updated_at"],
                    json.dumps(task, ensure_ascii=False),
                ),
            )
            return task, True

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM orchestration_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def save(self, task: dict[str, Any], expected_revision: int) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE orchestration_tasks
                SET status = ?, revision = ?, updated_at = ?, payload = ?
                WHERE id = ? AND revision = ?
                """,
                (
                    task["status"],
                    task["revision"],
                    task["updated_at"],
                    json.dumps(task, ensure_ascii=False),
                    task["id"],
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "task changed concurrently; read current state before retrying"
                )
