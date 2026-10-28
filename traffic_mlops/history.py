"""Small persistent SQLite store for forecasts served by the application."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from traffic_mlops.config import PROJECT_ROOT


DB_PATH = Path(os.getenv("TRAFFIC_APP_DB", str(PROJECT_ROOT / "traffic_app.db")))


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


def log_incident(junction: str, category: str, severity: str, summary: str, image_name: str | None = None) -> int:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            "INSERT INTO incident_reports(created_at,junction,category,severity,summary,image_name) VALUES(?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), junction, category, severity, summary, image_name),
        )
        return int(cursor.lastrowid)


def recent_incidents(limit: int = 100) -> list[dict[str, object]]:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id,created_at,junction,category,severity,summary,image_name "
            "FROM incident_reports ORDER BY id DESC LIMIT ?", (max(1, min(limit, 500)),)
        ).fetchall()
    return [dict(row) for row in rows]


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


def recent_virtual_alerts(limit: int = 100) -> list[dict[str, object]]:
    initialize_history()
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id,tick,created_at,junction,severity,alert FROM virtual_alerts ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 1000)),),
        ).fetchall()
    return [dict(row) for row in rows]
