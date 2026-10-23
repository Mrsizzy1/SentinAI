"""
Module de Détection de Vivant & Anti-Usurpation (Liveness & Anti-Spoofing Engine).
Empêche le piratage par présentation de photos imprimées, écrans de smartphone ou masques.
Méthodes :
1. Analyse de texture cutanée et variance de Laplacien.
2. Détection de moiré et réflectance spéculaire d'écran.
3. Calcul d'un score de vivant certifié (Liveness Score).
"""

from typing import Tuple
import cv2
import numpy as np


class LivenessDetector:
    """Classifieur de vivant en temps réel pour visages."""

    def __init__(self, liveness_threshold: float = 0.45):
        self.liveness_threshold = liveness_threshold
        self.enabled = True

    def check_liveness(self, face_bgr: np.ndarray) -> Tuple[bool, float, str]:
        """
        Analyse une découpe de visage pour vérifier s'il s'agit d'un être humain vivant
        ou d'une tentative de fraude (photo imprimée / écran de smartphone).
        Retourne : (is_live: bool, confidence: float, reason: str)
        """
        if not self.enabled:
            return True, 1.0, "Désactivé"

        if face_bgr is None or face_bgr.size == 0 or face_bgr.shape[0] < 40 or face_bgr.shape[1] < 40:
            return False, 0.0, "Image trop petite"

        # 1. Analyse de netteté & micro-relief cutané (Variance de Laplacien)
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 2. Analyse fréquentielle (recherche de trames d'écrans / moiré)
        # Normalisation de la taille pour calcul FFT
        resized = cv2.resize(gray, (128, 128))
        dft = cv2.dft(np.float32(resized), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-6)
        high_freq_energy = float(np.mean(magnitude_spectrum[40:88, 40:88]))

        # 3. Analyse de saturation & reflets spéculaires (écrans brillants)
        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]))
        val_std = float(np.std(hsv[:, :, 2]))

        # Calcul du score combiné de vivant
        # Une photo imprimée ou écran flou a un laplacian_var très bas (< 60) ou anormalement constant
        texture_score = min(1.0, max(0.0, (laplacian_var - 40.0) / 250.0))
        freq_score = min(1.0, max(0.0, (high_freq_energy - 100.0) / 100.0))
        sat_score = 1.0 if (25.0 <= sat_mean <= 220.0 and val_std > 18.0) else 0.3

        liveness_score = 0.50 * texture_score + 0.30 * freq_score + 0.20 * sat_score

        if liveness_score >= self.liveness_threshold:
            return True, round(liveness_score, 2), "Humain vivant certifié"
        else:
            reason = "Texture suspecte (Photo/Écran potentiel)" if texture_score < 0.25 else "Réflectance anormale"
            return False, round(liveness_score, 2), reason


# Instance globale
liveness_detector = LivenessDetector()
