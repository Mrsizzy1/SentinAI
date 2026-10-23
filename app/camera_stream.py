"""
Module de capture vidéo OpenCV locale pour la webcam du PC (Windows & Multi-plateforme).
Implémente un thread d'acquisition d'images dédié qui vide en permanence
le tampon matériel OpenCV pour garantir une latence minimale (zéro décalage temporel).

Supporte la détection intelligente des backends (MSMF vs DirectShow),
la gestion du format MJPG et la compatibilité spécifique avec les webcams HP HD Camera.
"""

import threading
import time
from typing import Any, Optional, Tuple, Union
import cv2
import numpy as np


class PCCameraStream:
    """Gestionnaire de flux vidéo OpenCV robuste pour webcam PC."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.backend_name = "Auto"

        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.is_opened = False
        self.current_backend_index = 0  # 0: MSMF, 1: DSHOW, 2: DEFAULT

    def _open_capture(self, index: Any, preferred_backend: Optional[int] = None) -> Optional[cv2.VideoCapture]:
        """Tente d'ouvrir la caméra (webcam locale ou flux réseau RTSP/ONVIF)."""
        # Support flux réseau IP / RTSP / FFMPEG
        if isinstance(index, str) and (index.startswith("rtsp://") or index.startswith("http://") or index.startswith("https://") or index.endswith((".mp4", ".mkv", ".avi"))):
            try:
                cap = cv2.VideoCapture(index, cv2.CAP_FFMPEG) if hasattr(cv2, "CAP_FFMPEG") else cv2.VideoCapture(index)
                if cap and cap.isOpened():
                    self.backend_name = "RTSP/FFMPEG"
                    print(f"[Camera] Flux réseau IP/RTSP initialisé avec succès : {index}")
                    return cap
            except Exception as e:
                print(f"[Camera] Échec connexion RTSP {index}: {e}")
            return None

        # Caméra locale (index int)
        try:
            cam_idx = int(index)
        except Exception:
            cam_idx = 0

        backends_to_try = []

        if preferred_backend is not None:
            backends_to_try.append(preferred_backend)
        else:
            # Ordre de préférence sur Windows :
            # 1. MSMF (Media Foundation - standard moderne Windows 10/11, optimal pour HP HD Camera)
            # 2. DSHOW (DirectShow)
            # 3. Default
            if hasattr(cv2, "CAP_MSMF"):
                backends_to_try.append(cv2.CAP_MSMF)
            if hasattr(cv2, "CAP_DSHOW"):
                backends_to_try.append(cv2.CAP_DSHOW)
            backends_to_try.append(None)

        for backend in backends_to_try:
            b_label = "MSMF" if backend == getattr(cv2, "CAP_MSMF", -1) else ("DSHOW" if backend == getattr(cv2, "CAP_DSHOW", -2) else "DEFAULT")
            try:
                cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
                if not cap or not cap.isOpened():
                    continue

                # Configuration codec et résolution
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except Exception:
                    pass

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, 30)

                # Test de lecture de frames pour vérifier si l'image n'est pas noire
                valid_frame = False
                for _ in range(5):
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        valid_frame = True
                        mean_b = float(np.mean(test_frame))
                        if mean_b > 1.0:
                            # Flux non-noir parfait !
                            print(f"[Camera] Succès d'ouverture Caméra #{index} via {b_label} (luminosité: {mean_b:.1f})")
                            self.backend_name = b_label
                            return cap

                if valid_frame:
                    # Même si l'image est actuellement sombre (ex: cache fermé ou faible éclairage),
                    # le flux matériel répond bien
                    print(f"[Camera] Caméra #{index} initialisée via {b_label}")
                    self.backend_name = b_label
                    return cap

                cap.release()
            except Exception as e:
                print(f"[Camera] Échec backend {b_label}: {e}")

        # Dernier essai sans forcer de résolution ni de codec
        try:
            cap = cv2.VideoCapture(index)
            if cap and cap.isOpened():
                self.backend_name = "DEFAULT"
                return cap
        except Exception:
            pass

        return None

    def start(self, preferred_backend: Optional[int] = None) -> bool:
        """Démarre la capture vidéo en arrière-plan."""
        if self._running:
            return True

        print(f"[Camera] Tentative d'ouverture de la webcam PC index #{self.camera_index}...")
        self.cap = self._open_capture(self.camera_index, preferred_backend)

        if not self.cap or not self.cap.isOpened():
            print(f"[Camera] Impossible d'ouvrir la caméra #{self.camera_index}.")
            self.is_opened = False
            return False

        self.is_opened = True
        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()
        print(f"[Camera] Webcam PC #{self.camera_index} prête ({self.backend_name}).")
        return True

    def _capture_worker(self) -> None:
        """Boucle d'acquisition continue pour vider le tampon matériel."""
        while self._running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                # Si la caméra fournit une résolution différente, redimensionner proprement
                h, w = frame.shape[:2]
                if w != self.width or h != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

                with self._lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Récupère l'image la plus récente de manière instantanée."""
        with self._lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def set_camera_index(self, new_index: int) -> bool:
        """Change dynamiquement l'index de la caméra avec repli automatique sur la caméra précédente."""
        return self.set_source(new_index)

    def set_source(self, source: Any) -> bool:
        """Change la source vidéo (index webcam local ou URL réseau RTSP/ONVIF)."""
        prev_source = self.camera_index
        self.stop()
        self.camera_index = source
        success = self.start()
        if not success:
            print(f"[Camera] Source '{source}' indisponible. Rétablissement de '{prev_source}'...")
            self.camera_index = prev_source
            return self.start()
        return True

    def cycle_backend(self) -> bool:
        """Alterne entre MSMF et DirectShow."""
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, None]
        self.current_backend_index = (self.current_backend_index + 1) % len(backends)
        chosen = backends[self.current_backend_index]
        self.stop()
        return self.start(preferred_backend=chosen)

    def stop(self) -> None:
        """Arrête proprement le flux et libère le périphérique."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_opened = False
        self.latest_frame = None
        print(f"[Camera] Webcam #{self.camera_index} libérée.")
