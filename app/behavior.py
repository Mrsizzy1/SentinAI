"""
Module d'Analyse Comportementale & Sûreté Industrielle (Behavior & Anomaly Engine).
Fonctionnalités Entreprise :
1. Zone Interdite / Geofence Virtuelle (Détection d'intrusion en zone sensible).
2. Détection de Stationnement Prolongé / Errance (Loitering detection).
3. Détection de Chute de Personne / Malaise (Man-down & Safety detection).
4. Détection de Mouvements Brusques / Course / Panique (Running / Fast movement).
5. Contrôle d'Accès & Alertes en Temps Réel.
"""

from datetime import datetime
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.actions import action_dispatcher


class BehaviorEngine:
    """Moteur d'analyse comportementale et de détection d'anomalies en temps réel."""

    def __init__(
        self,
        restricted_zone_enabled: bool = True,
        restricted_zone: Tuple[float, float, float, float] = (0.58, 0.15, 0.98, 0.88),
        loitering_threshold_s: float = 6.0,
        fall_detection_enabled: bool = True,
        fast_movement_enabled: bool = True,
    ):
        self.restricted_zone_enabled = restricted_zone_enabled
        # Zone normalisée [x1, y1, x2, y2] en pourcentage de l'image (0.0 à 1.0)
        self.restricted_zone = list(restricted_zone)
        self.loitering_threshold_s = loitering_threshold_s
        self.fall_detection_enabled = fall_detection_enabled
        self.fast_movement_enabled = fast_movement_enabled

        # Suivi des personnes dans le champ : id -> {first_seen, last_seen, last_box, last_alert_time}
        self._tracked_subjects: Dict[int, Dict[str, Any]] = {}
        self._next_track_id = 1
        self._alert_cooldowns: Dict[str, float] = {}  # type_alert -> timestamp

    def update_config(self, config: Dict[str, Any]) -> None:
        """Met à jour les seuils de sûreté à la volée."""
        if "restricted_zone_enabled" in config:
            self.restricted_zone_enabled = bool(config["restricted_zone_enabled"])
        if "restricted_zone" in config and len(config["restricted_zone"]) == 4:
            self.restricted_zone = [float(v) for v in config["restricted_zone"]]
        if "loitering_threshold_s" in config:
            self.loitering_threshold_s = float(config["loitering_threshold_s"])
        if "fall_detection_enabled" in config:
            self.fall_detection_enabled = bool(config["fall_detection_enabled"])
        if "fast_movement_enabled" in config:
            self.fast_movement_enabled = bool(config["fast_movement_enabled"])

    def _match_or_create_track(self, box: List[int], now: float) -> int:
        """Associe une détection à un sujet suivi (suivi centroïde simple et rapide)."""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        best_id = None
        min_dist = 110.0  # Seuil de distance maximale en pixels pour le même individu

        # Nettoyage des sujets disparus depuis plus de 3.0 secondes
        stale_ids = [tid for tid, s in self._tracked_subjects.items() if now - s["last_seen"] > 3.0]
        for tid in stale_ids:
            del self._tracked_subjects[tid]

        for tid, s in self._tracked_subjects.items():
            lx1, ly1, lx2, ly2 = s["last_box"]
            lcx, lcy = (lx1 + lx2) // 2, (ly1 + ly2) // 2
            dist = np.hypot(cx - lcx, cy - lcy)
            if dist < min_dist:
                min_dist = dist
                best_id = tid

        if best_id is not None:
            # Calcul de la vélocité
            s = self._tracked_subjects[best_id]
            dt = max(0.01, now - s["last_seen"])
            s["speed_px_per_s"] = min_dist / dt
            s["last_seen"] = now
            s["last_box"] = box
            return best_id
        else:
            new_id = self._next_track_id
            self._next_track_id += 1
            self._tracked_subjects[new_id] = {
                "first_seen": now,
                "last_seen": now,
                "last_box": box,
                "speed_px_per_s": 0.0,
            }
            return new_id

    def analyze(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        audit_logger: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Analyse l'ensemble des comportements sur l'image courante.
        Retourne la liste des alertes actives et un dictionnaire de synthèse.
        """
        h, w = frame.shape[:2]
        now = time.time()
        active_alerts: List[Dict[str, Any]] = []

        # Coordonnées réelles en pixels de la zone restreinte
        rz_x1 = int(self.restricted_zone[0] * w)
        rz_y1 = int(self.restricted_zone[1] * h)
        rz_x2 = int(self.restricted_zone[2] * w)
        rz_y2 = int(self.restricted_zone[3] * h)

        humans = [d for d in detections if d.get("category") == "human"]

        for det in humans:
            x1, y1, x2, y2 = det["box"]
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 1. Suivi temporel
            track_id = self._match_or_create_track(det["box"], now)
            track_info = self._tracked_subjects[track_id]
            dwell_time = now - track_info["first_seen"]
            det["dwell_time_s"] = round(dwell_time, 1)
            det["track_id"] = track_id

            face_name = det.get("face_name", "Inconnu")
            spoof_detected = det.get("spoof_detected", False)

            # RÈGLE 0 : Alerte Fraude / Tentative d'usurpation biométrique (Anti-Spoofing)
            if spoof_detected:
                spoof_desc = "ALERTE SÛRETÉ : Tentative d'usurpation biométrique (Photo ou écran détecté)"
                spoof_alert = {
                    "type": "SPOOF_ATTEMPT",
                    "severity": "CRITICAL",
                    "person_name": "Fraudeur / Non-Vivant",
                    "box": det["box"],
                    "description": spoof_desc,
                }
                active_alerts.append(spoof_alert)
                cooldown_key = f"SPOOF_{track_id}"
                if now - self._alert_cooldowns.get(cooldown_key, 0.0) > 8.0:
                    self._alert_cooldowns[cooldown_key] = now
                    if audit_logger:
                        audit_logger.log_event(
                            event_type="SPOOF_ATTEMPT",
                            severity="CRITICAL",
                            person_name="Fraudeur (Photo)",
                            description=spoof_desc,
                            frame=frame,
                        )
                    action_dispatcher.trigger_incident_webhook(spoof_alert)

            # Contrôle d'accès automatique : Déverrouillage de porte si identité autorisée et vivante certifiée
            if face_name != "Inconnu" and not spoof_detected and det.get("face_conf", 0) > 0.60:
                action_dispatcher.trigger_door_relay("UNLOCK", duration_s=4.0)

            # 2. RÈGLE 1 : Intrusion en Zone Restreinte / Périmètre Interdit
            if self.restricted_zone_enabled:
                in_zone = (rz_x1 <= cx <= rz_x2) and (rz_y1 <= cy <= rz_y2)
                if in_zone:
                    det["in_restricted_zone"] = True
                    is_unauthorized = (face_name == "Inconnu")
                    severity = "CRITICAL" if is_unauthorized else "INFO"
                    alert_type = "ZONE_INTRUSION"
                    desc = (
                        f"INTRUSION ZONE RESTREINTE : Individu non autorisé ({int(det['confidence']*100)}%)"
                        if is_unauthorized
                        else f"Présence en zone restreinte : {face_name}"
                    )

                    alert = {
                        "type": alert_type,
                        "severity": severity,
                        "person_name": face_name,
                        "box": det["box"],
                        "description": desc,
                    }
                    active_alerts.append(alert)

                    # Journalisation avec cooldown de 6 secondes pour éviter le flood
                    cooldown_key = f"{alert_type}_{track_id}"
                    if now - self._alert_cooldowns.get(cooldown_key, 0.0) > 6.0:
                        self._alert_cooldowns[cooldown_key] = now
                        if audit_logger:
                            audit_logger.log_event(
                                event_type=alert_type,
                                severity=severity,
                                person_name=face_name,
                                description=desc,
                                frame=frame if severity == "CRITICAL" else None,
                            )
                        if severity == "CRITICAL":
                            action_dispatcher.trigger_incident_webhook(alert)

            # 3. RÈGLE 2 : Détection de Chute / Malaise (Man-down safety)
            if self.fall_detection_enabled:
                aspect_ratio = bw / float(bh)
                # Une personne debout a un ratio h/w typique > 1.5 (donc w/h < 0.65).
                # Une personne au sol a un ratio horizontal w/h > 1.25 et se trouve vers le bas de la scène.
                is_fallen = (aspect_ratio >= 1.25) and (bh >= 35) and (cy > h * 0.40)
                if is_fallen:
                    det["behavior_alert"] = "CHUTE"
                    desc = f"URGENCE SÉCURITÉ : Chute de personne détectée ({face_name})"
                    fall_alert = {
                        "type": "FALL_DETECTED",
                        "severity": "CRITICAL",
                        "person_name": face_name,
                        "box": det["box"],
                        "description": desc,
                    }
                    active_alerts.append(fall_alert)

                    cooldown_key = f"FALL_{track_id}"
                    if now - self._alert_cooldowns.get(cooldown_key, 0.0) > 8.0:
                        self._alert_cooldowns[cooldown_key] = now
                        if audit_logger:
                            audit_logger.log_event(
                                event_type="FALL_DETECTED",
                                severity="CRITICAL",
                                person_name=face_name,
                                description=desc,
                                frame=frame,
                            )
                        action_dispatcher.trigger_incident_webhook(fall_alert)

            # 4. RÈGLE 3 : Détection de Stationnement Prolongé (Errance / Loitering)
            if dwell_time >= self.loitering_threshold_s:
                desc = f"Stationnement prolongé / Errance ({int(dwell_time)}s) - {face_name}"
                active_alerts.append({
                    "type": "LOITERING",
                    "severity": "WARNING",
                    "person_name": face_name,
                    "box": det["box"],
                    "description": desc,
                })

                cooldown_key = f"LOITERING_{track_id}"
                if now - self._alert_cooldowns.get(cooldown_key, 0.0) > 12.0:
                    self._alert_cooldowns[cooldown_key] = now
                    if audit_logger:
                        audit_logger.log_event(
                            event_type="LOITERING_ALERT",
                            severity="WARNING",
                            person_name=face_name,
                            description=desc,
                        )

            # 5. RÈGLE 4 : Mouvement Brusque / Course / Panique
            if self.fast_movement_enabled:
                speed = track_info.get("speed_px_per_s", 0.0)
                if speed > (w * 0.50):  # Dépasse 50% de la largeur de l'écran en 1 seconde
                    desc = f"Mouvement brusque / Course anormale ({face_name})"
                    active_alerts.append({
                        "type": "FAST_MOVEMENT",
                        "severity": "WARNING",
                        "person_name": face_name,
                        "box": det["box"],
                        "description": desc,
                    })

        summary = {
            "restricted_zone_active": self.restricted_zone_enabled,
            "restricted_zone_coords": [rz_x1, rz_y1, rz_x2, rz_y2],
            "active_alerts_count": len(active_alerts),
            "critical_count": sum(1 for a in active_alerts if a["severity"] == "CRITICAL"),
            "warning_count": sum(1 for a in active_alerts if a["severity"] == "WARNING"),
            "tracked_count": len(humans),
        }

        return active_alerts, summary


# Instance globale
behavior_engine = BehaviorEngine()
