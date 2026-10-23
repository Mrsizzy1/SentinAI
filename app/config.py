"""
Module de gestion et de validation de la configuration (config.yaml).
Supporte le chargement, les valeurs par défaut et la mise à jour dynamique.
"""

from pathlib import Path
import threading
from typing import Any, Dict, List
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "server": {
        "host": "0.0.0.0",
        "port": 8443,
        "ssl_enabled": True,
        "cert_file": "certs/cert.pem",
        "key_file": "certs/key.pem",
    },
    "camera": {
        "default_mode": "browser",
        "pc_camera_index": 0,
        "capture_width": 640,
        "capture_height": 480,
        "target_fps": 30,
    },
    "yolo": {
        "model_name": "yolov8n.pt",
        "confidence_threshold": 0.45,
        "iou_threshold": 0.45,
        "img_size": 640,
        "open_vocabulary_enabled": False,
        "open_vocabulary_model": "yolov8s-world.pt",
        "custom_classes": [
            "personne", "chat", "chien", "sac à dos",
            "smartphone", "clé", "bouteille", "chaussure"
        ],
    },
    "faces": {
        "enabled": True,
        "tolerance": 0.52,
        "storage_dir": "data/faces",
        "min_face_size": 40,
    },
    "cloud_fallback": {
        "enabled": False,
        "provider": "google",
        "api_key_or_creds": "",
        "trigger_confidence_threshold": 0.35,
    },
    "storage": {
        "captures_dir": "data/captures",
    },
}


class ConfigManager:
    """Gestionnaire thread-safe de la configuration."""

    def __init__(self, path: Path = CONFIG_PATH):
        self.path = path
        self._lock = threading.RLock()
        self.config: Dict[str, Any] = {}
        self.load()

    def load(self) -> Dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                self.config = DEFAULT_CONFIG.copy()
                self.save()
                return self.config

            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                # Fusion avec les valeurs par défaut
                self.config = self._deep_merge(DEFAULT_CONFIG.copy(), loaded)
            except Exception as e:
                print(f"[Config] Erreur de lecture de {self.path} ({e}), utilisation des valeurs par défaut.")
                self.config = DEFAULT_CONFIG.copy()
            return self.config

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    def get(self, *keys: str, default: Any = None) -> Any:
        with self._lock:
            curr: Any = self.config
            for k in keys:
                if isinstance(curr, dict) and k in curr:
                    curr = curr[k]
                else:
                    return default
            return curr

    def set(self, *keys_and_val: Any) -> None:
        """Modifie une clé imbriquée: set('yolo', 'confidence_threshold', 0.5)"""
        if len(keys_and_val) < 2:
            return
        keys = keys_and_val[:-1]
        val = keys_and_val[-1]

        with self._lock:
            curr = self.config
            for k in keys[:-1]:
                if k not in curr or not isinstance(curr[k], dict):
                    curr[k] = {}
                curr = curr[k]
            curr[keys[-1]] = val
            self.save()

    def _deep_merge(self, base: dict, overlay: dict) -> dict:
        for k, v in overlay.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                base[k] = self._deep_merge(base[k], v)
            else:
                base[k] = v
        return base


# Instance globale
cfg = ConfigManager()
