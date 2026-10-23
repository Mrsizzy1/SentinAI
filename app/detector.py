"""
Moteur de détection d'objets, personnes et animaux basé sur Ultralytics YOLOv8 & YOLO-World.
Supporte :
- Inférence multi-device (CUDA / Apple MPS / CPU avec repli automatique)
- YOLOv8 standard (yolov8n, yolov8s, yolov8m, etc.)
- Vocabulaire ouvert temps réel avec YOLO-World (set_classes sans réentraînement)
- Traçage esthétique des boîtes englobantes et badges d'information
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Catégories COCO 80 pour classification simplifiée
ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
}
HUMAN_CLASSES = {"person"}


def detect_optimal_device() -> str:
    """Détecte le matériel le plus performant disponible."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] GPU NVIDIA détecté : {gpu_name} -> Utilisation de CUDA")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Device] Apple Silicon GPU détecté -> Utilisation de MPS")
        return "mps"
    else:
        print("[Device] Aucun GPU accéléré trouvé -> Utilisation du CPU")
        return "cpu"


class ObjectDetector:
    """Moteur unifié de détection YOLOv8 et YOLO-World."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        open_vocab_enabled: bool = False,
        open_vocab_model: str = "yolov8s-world.pt",
        custom_classes: Optional[List[str]] = None,
    ):
        self.device = detect_optimal_device()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.model_name = model_name
        self.open_vocab_model_name = open_vocab_model
        self.open_vocab_enabled = open_vocab_enabled
        self.custom_classes = custom_classes or ["personne", "chat", "sac à dos", "téléphone"]

        self._active_model_name = ""
        self.model: Optional[YOLO] = None
        self._load_model()

        # Statistiques
        self.last_inference_time_ms: float = 0.0
        self._fps_history: List[float] = []

    def _load_model(self) -> None:
        """Charge ou recharge le modèle sélectionné."""
        target_model = self.open_vocab_model_name if self.open_vocab_enabled else self.model_name
        if target_model != self._active_model_name or self.model is None:
            print(f"[Detector] Chargement du modèle {target_model} sur {self.device}...")
            self.model = YOLO(target_model)
            self._active_model_name = target_model

            if self.open_vocab_enabled and hasattr(self.model, "set_classes"):
                print(f"[Detector] YOLO-World: Application des classes personnalisées : {self.custom_classes}")
                self.model.set_classes(self.custom_classes)

    def set_model(self, model_name: str) -> None:
        """Change le modèle standard (yolov8n.pt, yolov8s.pt...)."""
        if self.model_name != model_name:
            self.model_name = model_name
            if not self.open_vocab_enabled:
                self._load_model()

    def set_open_vocabulary(self, enabled: bool, classes: Optional[List[str]] = None) -> None:
        """Active ou désactive le mode vocabulaire ouvert YOLO-World."""
        self.open_vocab_enabled = enabled
        if classes:
            self.custom_classes = classes

        self._load_model()
        if self.open_vocab_enabled and self.model and hasattr(self.model, "set_classes"):
            self.model.set_classes(self.custom_classes)

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Effectue l'inférence sur l'image passée en paramètre.
        Retourne la liste des détections et le temps d'inférence en millisecondes.
        """
        if self.model is None:
            return [], 0.0

        t0 = time.perf_counter()

        # Inférence avec Ultralytics
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        t1 = time.perf_counter()
        inference_time_ms = (t1 - t0) * 1000.0
        self.last_inference_time_ms = inference_time_ms

        detections: List[Dict[str, Any]] = []
        if not results:
            return detections, inference_time_ms

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections, inference_time_ms

        names = self.model.names if hasattr(self.model, "names") else {}

        # Traitement des prédictions
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            if isinstance(names, dict):
                label = names.get(cls_id, str(cls_id))
            elif isinstance(names, list) and cls_id < len(names):
                label = names[cls_id]
            else:
                label = str(cls_id)

            label_lower = label.lower()
            is_human = (label_lower in HUMAN_CLASSES or "person" in label_lower or "humain" in label_lower)
            is_animal = (label_lower in ANIMAL_CLASSES or any(a in label_lower for a in ["chat", "chien", "cheval", "oiseau"]))

            category = "human" if is_human else ("animal" if is_animal else "object")

            detections.append({
                "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                "label": label,
                "confidence": round(conf, 3),
                "category": category,
            })

        return detections, inference_time_ms

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        fps: float = 0.0,
        behavior_summary: Optional[Dict[str, Any]] = None,
        active_alerts: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Dessine les cadres modernes, zones de sûreté et alertes comportementales.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Dessin de la Zone de Sécurité Restreinte (si active)
        if behavior_summary and behavior_summary.get("restricted_zone_active"):
            rz = behavior_summary.get("restricted_zone_coords")
            if rz and len(rz) == 4:
                rx1, ry1, rx2, ry2 = rz
                rx1, ry1 = max(0, rx1), max(0, ry1)
                rx2, ry2 = min(w, rx2), min(h, ry2)
                if rx2 > rx1 and ry2 > ry1:
                    sub = annotated[ry1:ry2, rx1:rx2]
                    rect_overlay = sub.copy()
                    cv2.rectangle(rect_overlay, (0, 0), (rx2 - rx1, ry2 - ry1), (25, 25, 180), -1)
                    cv2.addWeighted(rect_overlay, 0.20, sub, 0.80, 0, sub)
                    annotated[ry1:ry2, rx1:rx2] = sub
                    cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (50, 50, 235), 2)
                    cv2.putText(
                        annotated,
                        "ZONE RESTREINTE / PERIMETRE SÉCURISÉ",
                        (rx1 + 10, ry1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (60, 60, 245),
                        1,
                        cv2.LINE_AA,
                    )

        color_map = {
            "human": (210, 70, 220),
            "animal": (40, 190, 255),
            "object": (230, 180, 30),
            "unknown": (180, 180, 180),
        }

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cat = det.get("category", "object")
            color = color_map.get(cat, (230, 180, 30))

            # Si la personne est en zone restreinte ou en alerte de chute
            in_restricted = det.get("in_restricted_zone", False)
            is_fallen = (det.get("behavior_alert") == "CHUTE")
            dwell_s = det.get("dwell_time_s")

            # Si une identification faciale est fournie
            face_name = det.get("face_name")
            face_desc = det.get("face_desc")
            if face_name:
                if face_name == "Inconnu":
                    label_text = f"Humain : Inconnu ({int(det['confidence']*100)}%)"
                    sub_text = "[Web Lens : 'W']"
                    color = (50, 50, 235) if in_restricted else (70, 70, 220)
                else:
                    conf_pct = int(det.get('face_conf', det['confidence'])*100)
                    label_text = f"ID: {face_name} ({conf_pct}%)"
                    sub_text = face_desc[:32] + ".." if face_desc and len(face_desc) > 32 else (face_desc or "")
                    color = (60, 220, 110)
            else:
                label_text = f"{det['label']} {int(det['confidence']*100)}%"
                sub_text = ""

            if is_fallen:
                sub_text = "⚠️ CHUTE DÉTECTÉE"
                color = (30, 30, 240)
            elif in_restricted and face_name == "Inconnu":
                sub_text = "🚨 INTRUSION ZONE RESTREINTE"
                color = (30, 30, 240)
            elif dwell_s is not None and dwell_s >= 4.0:
                sub_text += f" [Présence: {int(dwell_s)}s]"

            # Cadre épuré avec coins accentués
            border_thick = 3 if (in_restricted or is_fallen) else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, border_thick)
            corner_len = min(18, (x2 - x1) // 3, (y2 - y1) // 3)
            thick = 3
            # 4 coins stylisés
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, thick)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, thick)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, thick)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, thick)
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, thick)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, thick)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, thick)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, thick)

            # Badge texte avec fond et description d'identité
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label_text, font, 0.52, 1)
            total_h = th + 8
            if sub_text:
                (sw, sh), _ = cv2.getTextSize(sub_text, font, 0.38, 1)
                box_w = max(tw, sw) + 12
                total_h += sh + 6
            else:
                box_w = tw + 10

            badge_y1 = max(0, y1 - total_h - 2)
            badge_y2 = y1
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + box_w, badge_y2), color, -1)
            cv2.putText(
                annotated,
                label_text,
                (x1 + 5, badge_y1 + th + 4),
                font,
                0.52,
                (15, 15, 20),
                1,
                cv2.LINE_AA,
            )
            if sub_text:
                cv2.putText(
                    annotated,
                    sub_text,
                    (x1 + 5, badge_y2 - 3),
                    font,
                    0.38,
                    (25, 35, 30) if face_name not in ("Inconnu", None) and not in_restricted else (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 2. Bannière supérieure pour alertes critiques actives
        if active_alerts:
            alert = active_alerts[0]
            banner_color = (20, 20, 210) if alert["severity"] == "CRITICAL" else (20, 140, 220)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 40), (w, 75), banner_color, -1)
            cv2.addWeighted(overlay, 0.92, annotated, 0.08, 0, annotated)
            cv2.putText(
                annotated,
                f"SURETE [{alert['severity']}] : {alert['description']}",
                (15, 63),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Affichage du bandeau supérieur (HUD : FPS, Détections, Modèle)
        hud_bg = annotated[0:40, 0:w]
        overlay = hud_bg.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, hud_bg, 0.25, 0, hud_bg)

        hud_text = f"FPS: {fps:.1f} | Latence: {self.last_inference_time_ms:.1f}ms | Objets: {len(detections)}"
        cv2.putText(
            annotated,
            hud_text,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 210),
            1,
            cv2.LINE_AA,
        )

        mode_badge = "YOLO-World [Vocab Ouvert]" if self.open_vocab_enabled else f"YOLOv8 [{self.model_name}]"
        (bw, _), _ = cv2.getTextSize(mode_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(
            annotated,
            mode_badge,
            (w - bw - 15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 255),
            1,
            cv2.LINE_AA,
        )

        return annotated
