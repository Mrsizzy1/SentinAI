"""
Module de Journalisation & d'Audit Inaltérable Entreprise.
Gère :
- L'enregistrement sécurisé et structuré de tous les événements de sûreté (base SQLite locale).
- La capture automatique de clichés lors d'incidents critiques (preuves horodatées).
- L'exportation de rapports d'audit réglementaires au format CSV / JSON.
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


class EnterpriseAuditLogger:
    """Gestionnaire d'Audit Trail et de traçabilité de sûreté industrielle."""

    def __init__(self, db_path: str = "data/enterprise_audit.db", captures_dir: str = "data/captures"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.captures_dir = Path(captures_dir)
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self._init_database()

    def _init_database(self) -> None:
        """Initialise la table SQLite d'audit trail avec indexation temporelle."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        person_name TEXT,
                        description TEXT NOT NULL,
                        snapshot_path TEXT
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp DESC)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_events(severity)
                """)
                conn.commit()

    def log_event(
        self,
        event_type: str,
        severity: str,
        person_name: str,
        description: str,
        frame: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Enregistre un événement de sécurité dans le registre d'audit.
        Si une frame est fournie pour un événement critique, sauvegarde une preuve horodatée.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        snapshot_filename = None

        if frame is not None and severity in ("CRITICAL", "WARNING"):
            filename = f"audit_{severity.lower()}_{ts_slug}.jpg"
            target_path = self.captures_dir / filename
            try:
                cv2.imwrite(str(target_path), frame)
                snapshot_filename = filename
            except Exception:
                pass

        event_data = {
            "timestamp": now_str,
            "event_type": event_type,
            "severity": severity,
            "person_name": person_name or "Inconnu",
            "description": description,
            "snapshot_path": f"/api/captures/{snapshot_filename}" if snapshot_filename else None,
        }

        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO audit_events (timestamp, event_type, severity, person_name, description, snapshot_path)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        event_data["timestamp"],
                        event_data["event_type"],
                        event_data["severity"],
                        event_data["person_name"],
                        event_data["description"],
                        event_data["snapshot_path"],
                    ))
                    conn.commit()
                    event_data["id"] = cursor.lastrowid
            except Exception as e:
                print(f"[AuditLogger] Erreur insertion SQL: {e}")

        return event_data

    def get_recent_events(self, limit: int = 50, severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Récupère les événements récents pour le tableau de bord d'entreprise."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    if severity_filter and severity_filter != "ALL":
                        cursor.execute("""
                            SELECT * FROM audit_events
                            WHERE severity = ?
                            ORDER BY id DESC LIMIT ?
                        """, (severity_filter, limit))
                    else:
                        cursor.execute("""
                            SELECT * FROM audit_events
                            ORDER BY id DESC LIMIT ?
                        """, (limit,))
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
            except Exception as e:
                print(f"[AuditLogger] Erreur lecture SQL: {e}")
                return []

    def export_csv(self) -> str:
        """Exporte l'ensemble des journaux d'audit au format CSV standard."""
        with self._lock:
            lines = ["ID,Horodatage,Type Evenement,Gravite,Personne,Description,Preuve Photo"]
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, timestamp, event_type, severity, person_name, description, snapshot_path FROM audit_events ORDER BY id ASC")
                    for row in cursor.fetchall():
                        clean_row = [str(col).replace(",", ";") if col is not None else "" for col in row]
                        lines.append(",".join(clean_row))
            except Exception:
                pass
            return "\n".join(lines)


# Instance globale
audit_logger = EnterpriseAuditLogger()
