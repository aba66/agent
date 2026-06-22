from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import TaskRecord, UserContext
from .utils import ensure_directory


class TaskStore:
    """Persist sessions, task runs, versions, and traces into SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).resolve()
        ensure_directory(self.db_path.parent)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    parent_task_id TEXT,
                    revision INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_text TEXT NOT NULL,
                    input_files_json TEXT NOT NULL,
                    template_file TEXT,
                    requirement_json TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL,
                    clarifications_json TEXT NOT NULL,
                    errors_json TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );
                """
            )

    def get_or_create_session(
        self,
        session_id: str | None,
        title: str,
        user: UserContext,
    ) -> str:
        now = datetime.utcnow().isoformat()
        with self._connect() as connection:
            if session_id:
                row = connection.execute(
                    "SELECT session_id, title FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row:
                    resolved_title = title.strip() if title and title.strip() else row["title"]
                    connection.execute(
                        "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                        (resolved_title, now, session_id),
                    )
                    connection.commit()
                    return session_id

            new_session_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO sessions (session_id, title, username, role, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (new_session_id, title, user.username, user.role, now, now),
            )
            connection.commit()
            return new_session_id

    def start_task(
        self,
        session_id: str,
        parent_task_id: str | None,
        title: str,
        user: UserContext,
        request_text: str,
        input_files: list[str],
        template_file: str | None,
    ) -> tuple[str, int]:
        now = datetime.utcnow().isoformat()
        task_id = str(uuid.uuid4())
        with self._connect() as connection:
            if parent_task_id:
                parent_row = connection.execute(
                    "SELECT revision FROM tasks WHERE task_id = ?",
                    (parent_task_id,),
                ).fetchone()
                revision = int(parent_row["revision"]) + 1 if parent_row else 1
            else:
                row = connection.execute(
                    "SELECT COALESCE(MAX(revision), 0) AS max_revision FROM tasks WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                revision = int(row["max_revision"]) + 1 if row else 1

            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, session_id, parent_task_id, revision, title,
                    username, role, status, request_text, input_files_json,
                    template_file, requirement_json, artifacts_json, clarifications_json,
                    errors_json, trace_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    session_id,
                    parent_task_id,
                    revision,
                    title,
                    user.username,
                    user.role,
                    "running",
                    request_text,
                    json.dumps(input_files, ensure_ascii=False),
                    template_file,
                    json.dumps({}, ensure_ascii=False),
                    json.dumps({}, ensure_ascii=False),
                    json.dumps([], ensure_ascii=False),
                    json.dumps([], ensure_ascii=False),
                    json.dumps([], ensure_ascii=False),
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            connection.commit()
        return task_id, revision

    def finish_task(
        self,
        task_id: str,
        status: str,
        requirement: dict[str, Any],
        artifacts: dict[str, Any],
        clarifications: list[str],
        errors: list[str],
        trace_events: list[dict[str, Any]],
    ) -> TaskRecord:
        now = datetime.utcnow().isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE tasks
                SET status = ?, requirement_json = ?, artifacts_json = ?,
                    clarifications_json = ?, errors_json = ?, trace_json = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    status,
                    json.dumps(requirement, ensure_ascii=False),
                    json.dumps(artifacts, ensure_ascii=False),
                    json.dumps(clarifications, ensure_ascii=False),
                    json.dumps(errors, ensure_ascii=False),
                    json.dumps(trace_events, ensure_ascii=False),
                    now,
                    task_id,
                ),
            )
            connection.commit()
            row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return self._row_to_task_record(row)

    def cancel_running_tasks(self, session_id: str, reason: str = "用户主动终止处理") -> int:
        now = datetime.utcnow().isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT task_id, errors_json FROM tasks WHERE session_id = ? AND status = ?",
                (session_id, "running"),
            ).fetchall()
            canceled = 0
            for row in rows:
                errors = json.loads(row["errors_json"] or "[]")
                if reason not in errors:
                    errors.append(reason)
                connection.execute(
                    """
                    UPDATE tasks
                    SET status = ?, errors_json = ?, updated_at = ?
                    WHERE task_id = ?
                    """,
                    ("canceled", json.dumps(errors, ensure_ascii=False), now, row["task_id"]),
                )
                canceled += 1
            if canceled:
                connection.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
            connection.commit()
        return canceled

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT session_id, title, username, role, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_tasks(self, session_id: str | None = None, limit: int = 50) -> list[TaskRecord]:
        with self._connect() as connection:
            if session_id:
                rows = connection.execute(
                    """
                    SELECT * FROM tasks
                    WHERE session_id = ?
                    ORDER BY revision DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM tasks
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [self._row_to_task_record(row) for row in rows]

    def get_latest_task(self, session_id: str) -> TaskRecord | None:
        tasks = self.list_tasks(session_id=session_id, limit=1)
        return tasks[0] if tasks else None

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return self._row_to_task_record(row) if row else None

    def get_task_lineage(self, task_id: str | None) -> list[TaskRecord]:
        if not task_id:
            return []

        lineage: list[TaskRecord] = []
        current_task_id = task_id
        visited: set[str] = set()
        while current_task_id and current_task_id not in visited:
            visited.add(current_task_id)
            task = self.get_task(current_task_id)
            if task is None:
                break
            lineage.append(task)
            current_task_id = task.parent_task_id
        lineage.reverse()
        return lineage

    def _row_to_task_record(self, row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=row["task_id"],
            session_id=row["session_id"],
            parent_task_id=row["parent_task_id"],
            revision=row["revision"],
            title=row["title"],
            username=row["username"],
            role=row["role"],
            status=row["status"],
            request_text=row["request_text"],
            input_files=json.loads(row["input_files_json"]),
            template_file=row["template_file"],
            requirement=json.loads(row["requirement_json"]),
            artifacts=json.loads(row["artifacts_json"]),
            clarifications=json.loads(row["clarifications_json"]),
            errors=json.loads(row["errors_json"]),
            trace_events=json.loads(row["trace_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
