"""
Module de Déclenchement d'Actions Automatisées Entreprise (Automated Response Engine).
Permet au système d'agir physiquement et de communiquer en direct ("Faire Tout") :
1. Contrôle d'accès physique (Ouverture/Verrouillage de portes, barrières).
2. Webhooks d'urgence vers les systèmes SIEM, Microsoft Teams, Slack ou serveurs centraux.
3. Déclenchement d'alarmes et de sirènes d'évacuation.
"""

from datetime import datetime
import json
import threading
from typing import Any, Dict, List, Optional
import requests


class EnterpriseActionDispatcher:
    """Gestionnaire d'actions automatisées en réponse aux alertes de sûreté."""

    def __init__(self):
        self._lock = threading.RLock()
        self.webhook_url: str = ""
        self.door_relay_status: str = "LOCKED"  # "LOCKED", "UNLOCKED", "LOCKDOWN"
        self.alarm_active: bool = False
        self.last_action_log: List[Dict[str, Any]] = []

    def set_webhook_url(self, url: str) -> None:
        """Configure l'URL de notification d'urgence externe."""
        with self._lock:
            self.webhook_url = url.strip()

    def trigger_door_relay(self, action: str = "UNLOCK", duration_s: float = 5.0) -> Dict[str, Any]:
        """
        Déclenche un relais de porte physique (ou barrière d'accès sécurisée).
        action: 'UNLOCK', 'LOCK', 'LOCKDOWN'
        """
        with self._lock:
            self.door_relay_status = "UNLOCKED" if action == "UNLOCK" else "LOCKDOWN"
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = f"Relais de porte commandé : {action} (Temporisation: {duration_s}s)"
            entry = {"timestamp": now_str, "action": f"DOOR_{action}", "status": "EXECUTED", "details": msg}
            self.last_action_log.append(entry)

            # Temporisation de reverrouillage automatique si UNLOCK
            if action == "UNLOCK":
                def auto_relock():
                    import time
                    time.sleep(duration_s)
                    with self._lock:
                        if self.door_relay_status == "UNLOCKED":
                            self.door_relay_status = "LOCKED"

                threading.Thread(target=auto_relock, daemon=True).start()

            return entry

    def trigger_incident_webhook(self, alert_data: Dict[str, Any]) -> None:
        """Envoie une notification d'alerte critique asynchrone vers un webhook externe."""
        if not self.webhook_url:
            return

        def _send():
            payload = {
                "source": "VisionAI-Enterprise-System",
                "timestamp": datetime.now().isoformat(),
                "alert": alert_data,
            }
            try:
                requests.post(self.webhook_url, json=payload, timeout=3.0)
            except Exception as e:
                print(f"[ActionDispatcher] Échec envoi Webhook ({self.webhook_url}): {e}")

        threading.Thread(target=_send, daemon=True).start()

    def get_status(self) -> Dict[str, Any]:
        """Renvoie l'état actuel des actionneurs."""
        with self._lock:
            return {
                "door_relay": self.door_relay_status,
                "alarm_active": self.alarm_active,
                "webhook_configured": bool(self.webhook_url),
                "webhook_url": self.webhook_url,
                "recent_actions": self.last_action_log[-10:],
            }


# Instance globale
action_dispatcher = EnterpriseActionDispatcher()
