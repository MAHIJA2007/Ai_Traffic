"""Small persistent SQLite store for forecasts served by the application."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from traffic_mlops.config import PROJECT_ROOT


DB_PATH = Path(os.getenv("TRAFFIC_APP_DB", str(PROJECT_ROOT / "traffic_app.db")))

DEFAULT_DEVICES = [
    ("adaptive-signal", "Adaptive signal controller", "signal", "on", "auto", "Junction-wide green phase"),
    ("ramp-meter", "Ramp meter", "flow", "off", "auto", "Metered on-ramp release"),
    ("vms", "Variable message sign", "signage", "on", "auto", "Traveler advisory display"),
    ("street-lighting", "Street lighting circuit", "lighting", "on", "auto", "Corridor illumination"),
    ("depot-barrier", "Depot access barrier", "access", "off", "manual", "Maintenance depot gate"),
]


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def initialize_history() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                junction TEXT NOT NULL,
                forecast_for TEXT NOT NULL,
                predicted_vehicles REAL NOT NULL,
                input_json TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS incident_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                junction TEXT NOT NULL,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                summary TEXT NOT NULL,
                image_name TEXT
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS operator_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                junction TEXT NOT NULL,
                action TEXT NOT NULL,
                note TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS virtual_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick INTEGER NOT NULL,
                recorded_at TEXT NOT NULL,
                junction TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS virtual_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                junction TEXT NOT NULL,
                severity TEXT NOT NULL,
                alert TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                state TEXT NOT NULL,
                mode TEXT NOT NULL,
                detail TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                updated_by TEXT NOT NULL
            )
        """)
        _ensure_column(connection, "virtual_alerts", "acknowledged", "acknowledged INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "incident_reports", "triage_json", "triage_json TEXT")
        if connection.execute("SELECT COUNT(*) FROM devices").fetchone()[0] == 0:
            now = datetime.now(timezone.utc).isoformat()
            connection.executemany(
                "INSERT INTO devices(slug,name,kind,state,mode,detail,updated_at,updated_by) VALUES(?,?,?,?,?,?,?,?)",
                [(*device, now, "system seed") for device in DEFAULT_DEVICES],
            )


def save_prediction(result: dict[str, object], inputs: dict[str, object]) -> int:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            "INSERT INTO predictions(created_at,junction,forecast_for,predicted_vehicles,input_json) VALUES(?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), result["junction"], result["forecast_for"],
             result["predicted_vehicles"], json.dumps(inputs)),
        )
        return int(cursor.lastrowid)


def recent_predictions(limit: int = 100) -> list[dict[str, object]]:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id,created_at,junction,forecast_for,predicted_vehicles,input_json "
            "FROM predictions ORDER BY id DESC LIMIT ?", (max(1, min(limit, 500)),)
        ).fetchall()
    return [dict(row) | {"inputs": json.loads(row["input_json"])} for row in rows]


def log_incident(junction: str, category: str, severity: str, summary: str, image_name: str | None = None,
                 triage: dict[str, object] | None = None) -> int:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            "INSERT INTO incident_reports(created_at,junction,category,severity,summary,image_name,triage_json) "
            "VALUES(?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), junction, category, severity, summary, image_name,
             json.dumps(triage) if triage else None),
        )
        return int(cursor.lastrowid)


def recent_incidents(limit: int = 100) -> list[dict[str, object]]:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id,created_at,junction,category,severity,summary,image_name,triage_json "
            "FROM incident_reports ORDER BY id DESC LIMIT ?", (max(1, min(limit, 500)),)
        ).fetchall()
    return [dict(row) | {"triage": json.loads(row["triage_json"]) if row["triage_json"] else None} for row in rows]


def log_operator_action(junction: str, action: str, note: str) -> int:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            "INSERT INTO operator_actions(created_at,junction,action,note) VALUES(?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), junction, action, note),
        )
        return int(cursor.lastrowid)


def recent_operator_actions(limit: int = 100) -> list[dict[str, object]]:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id,created_at,junction,action,note FROM operator_actions ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        ).fetchall()
    return [dict(row) for row in rows]


def save_virtual_tick(tick: int, readings: list[dict[str, object]], alerts: list[dict[str, object]]) -> None:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        for reading in readings:
            connection.execute(
                "INSERT INTO virtual_telemetry(tick,recorded_at,junction,payload_json) VALUES(?,?,?,?)",
                (tick, reading["recorded_at"], reading["junction"], json.dumps(reading)),
            )
        for item in alerts:
            connection.execute(
                "INSERT INTO virtual_alerts(tick,created_at,junction,severity,alert) VALUES(?,?,?,?,?)",
                (tick, item["created_at"], item["junction"], item["severity"], item["alert"]),
            )


def recent_virtual_readings(limit: int = 100) -> list[dict[str, object]]:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            "SELECT payload_json FROM virtual_telemetry ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 2000)),),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def recent_virtual_alerts(limit: int = 100, unacknowledged_only: bool = False) -> list[dict[str, object]]:
    initialize_history()
    clause = " WHERE acknowledged = 0" if unacknowledged_only else ""
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id,tick,created_at,junction,severity,alert,acknowledged FROM virtual_alerts"
            f"{clause} ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 1000)),),
        ).fetchall()
    return [dict(row) for row in rows]


def reset_virtual_history() -> None:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DELETE FROM virtual_telemetry")
        connection.execute("DELETE FROM virtual_alerts")


def acknowledge_alert(alert_id: int) -> bool:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute("UPDATE virtual_alerts SET acknowledged = 1 WHERE id = ?", (int(alert_id),))
        return cursor.rowcount > 0


def list_devices() -> list[dict[str, object]]:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id,slug,name,kind,state,mode,detail,updated_at,updated_by FROM devices ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def set_device_state(slug: str, *, state: str, mode: str, updated_by: str) -> dict[str, object] | None:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "UPDATE devices SET state = ?, mode = ?, updated_at = ?, updated_by = ? WHERE slug = ?",
            (state, mode, datetime.now(timezone.utc).isoformat(), updated_by, slug),
        )
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM devices WHERE slug = ?", (slug,)).fetchone()
    return dict(row) if row else None
