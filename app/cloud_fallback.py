"""
Module d'assistance Cloud Vision de secours (Google Cloud Vision / Azure AI Vision).
Désactivé par défaut. S'active uniquement si configuré dans config.yaml
lorsque le modèle local YOLO ne parvient pas à identifier un objet avec une confiance suffisante.
"""

import base64
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import requests


class CloudVisionFallback:
    """Passerelle vers les API Cloud Vision en mode secours."""

    def __init__(
        self,
        enabled: bool = False,
        provider: str = "google",
        api_key_or_creds: str = "",
        trigger_confidence_threshold: float = 0.35,
    ):
        self.enabled = enabled
        self.provider = provider.lower()
        self.api_key_or_creds = api_key_or_creds
        self.trigger_threshold = trigger_confidence_threshold

    def is_active(self) -> bool:
        return self.enabled and bool(self.api_key_or_creds.strip())

    def analyze_fallback(self, frame: np.ndarray) -> Optional[List[Dict[str, Any]]]:
        """
        Interroge le cloud si activé et qu'une clé valide est fournie.
        Retourne la liste des détections cloud ou None.
        """
        if not self.is_active():
            return None

        # Encode l'image en JPEG base64
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_img = base64.b64encode(buffer).decode("utf-8")

        if self.provider == "google":
            return self._call_google_vision(b64_img)
        elif self.provider == "azure":
            return self._call_azure_vision(buffer.tobytes())
        return None

    def _call_google_vision(self, b64_img: str) -> Optional[List[Dict[str, Any]]]:
        """Appel à l'API REST Google Cloud Vision."""
        url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key_or_creds}"
        payload = {
            "requests": [
                {
                    "image": {"content": b64_img},
                    "features": [
                        {"type": "OBJECT_LOCALIZATION", "maxResults": 10},
                        {"type": "LABEL_DETECTION", "maxResults": 5},
                    ],
                }
            ]
        }

        try:
            resp = requests.post(url, json=payload, timeout=4.0)
            if resp.status_code != 200:
                print(f"[CloudFallback] Erreur Google Vision {resp.status_code}: {resp.text}")
                return None

            data = resp.json()
            annotations = data.get("responses", [{}])[0].get("localizedObjectAnnotations", [])
            detections = []
            for obj in annotations:
                name = obj.get("name", "Objet Cloud")
                score = float(obj.get("score", 0.5))
                detections.append({
                    "label": f"[Cloud] {name}",
                    "confidence": round(score, 2),
                    "category": "object",
                    "cloud_source": "google",
                })
            return detections
        except Exception as e:
            print(f"[CloudFallback] Exception Google Vision : {e}")
            return None

    def _call_azure_vision(self, img_bytes: bytes) -> Optional[List[Dict[str, Any]]]:
        """Appel à l'API Azure AI Computer Vision."""
        endpoint = "https://vision-ai-fallback.cognitiveservices.azure.com/" # Remplaçable par l'endpoint utilisateur
        url = f"{endpoint}/vision/v3.2/detect"
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key_or_creds,
            "Content-Type": "application/octet-stream",
        }

        try:
            resp = requests.post(url, headers=headers, data=img_bytes, timeout=4.0)
            if resp.status_code != 200:
                return None
            data = resp.json()
            objects = data.get("objects", [])
            detections = []
            for obj in objects:
                name = obj.get("object", "Objet Cloud")
                conf = float(obj.get("confidence", 0.5))
                detections.append({
                    "label": f"[Azure] {name}",
                    "confidence": round(conf, 2),
                    "category": "object",
                    "cloud_source": "azure",
                })
            return detections
        except Exception as e:
            print(f"[CloudFallback] Exception Azure Vision : {e}")
            return None
