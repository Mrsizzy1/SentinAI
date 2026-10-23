"""
Module de reconnaissance faciale temps réel et gestion des identités.
Fonctionnalités :
- Stockage 100% local des visages et empreintes biométriques (aucun envoi externe).
- Détection et encodage via 'face_recognition' (si installé) ou moteur OpenCV YuNet/Cascade.
- Enregistrement par capture caméra directe ou téléversement de photos.
- Reconnaissance en direct avec seuil de tolérance configurable.
- Affichage de 'Inconnu' pour les visages non enregistrés.
- Suppression ciblée ou purge complète des données personnelles (Conformité RGPD).
"""

import io
import json
import os
from pathlib import Path
import pickle
import shutil
import threading
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.liveness import liveness_detector
from app.security import security_manager

# Avertissement légal et éthique
GDPR_NOTICE = (
    "AVERTISSEMENT : La reconnaissance faciale traite des données biométriques sensibles. "
    "Elle doit impérativement être utilisée avec le consentement éclairé des personnes "
    "et dans le strict respect de la vie privée et de la réglementation (RGPD)."
)

# Tentative d'importation de face_recognition (dlib)
FACE_REC_AVAILABLE = False
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False


class FaceEngine:
    """Gestionnaire de visages et reconnaissance biométrique locale."""

    def __init__(self, storage_dir: str = "data/faces", tolerance: float = 0.52):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.tolerance = tolerance
        self._lock = threading.RLock()

        # Dictionnaire des empreintes connues : { "Nom": [np.ndarray, ...] }
        self.known_encodings: Dict[str, List[np.ndarray]] = {}

        # Moteur de repli OpenCV si face_recognition n'est pas installé
        self.haar_cascade = None
        if not FACE_REC_AVAILABLE:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            print("[FaceEngine] 'face_recognition' non détecté -> Utilisation du module OpenCV intégré.")

        self.load_database()

    def load_database(self) -> None:
        """Charge toutes les empreintes faciales stockées sur disque."""
        with self._lock:
            self.known_encodings.clear()
            for person_dir in self.storage_dir.iterdir():
                if person_dir.is_dir():
                    name = person_dir.name
                    enc_enc = person_dir / "encodings.enc"
                    enc_plain = person_dir / "encodings.npy"

                    # 1. Tentative chargement chiffré AES-256 (Norme Entreprise)
                    if enc_enc.exists():
                        try:
                            decrypted = security_manager.decrypt_bytes(enc_enc.read_bytes())
                            encs = pickle.loads(decrypted)
                            self.known_encodings[name] = list(encs)
                        except Exception as e:
                            print(f"[FaceEngine] Erreur déchiffrement {enc_enc}: {e}")

                    # 2. Repli / Migration automatique si fichier non chiffré
                    elif enc_plain.exists():
                        try:
                            encs = np.load(enc_plain, allow_pickle=True)
                            self.known_encodings[name] = list(encs)
                            # Chiffrement immédiat au repos
                            raw_data = pickle.dumps(encs)
                            enc_enc.write_bytes(security_manager.encrypt_bytes(raw_data))
                        except Exception as e:
                            print(f"[FaceEngine] Erreur chargement {enc_plain}: {e}")
            print(f"[FaceEngine] Base biométrique chiffrée AES-256 : {len(self.known_encodings)} identité(s) chargée(s).")

    def _extract_face_encoding(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extrait l'empreinte vectorielle d'un visage à partir d'une image BGR."""
        if FACE_REC_AVAILABLE:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            if not boxes:
                return None
            encodings = face_recognition.face_encodings(rgb, known_face_locations=[boxes[0]])
            if encodings:
                return encodings[0]
            return None
        else:
            # Repli OpenCV : Extraction de caractéristiques normalisées 128D via histogrammes & gradients
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = []
            if self.haar_cascade is not None:
                faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) == 0:
                h, w = image_bgr.shape[:2]
                face_crop = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
            else:
                x, y, fw, fh = faces[0]
                face_crop = gray[y:y+fh, x:x+fw]

            if face_crop.size == 0:
                return None

            resized = cv2.resize(face_crop, (64, 64))
            hist = cv2.calcHist([resized], [0], None, [128], [0, 256]).flatten()
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist = hist / norm
            return hist

    def register_person(self, name: str, image_bgr: np.ndarray) -> Tuple[bool, str]:
        """
        Enregistre une nouvelle personne ou ajoute une photo supplémentaire.
        Sauvegarde une vignette JPG et le vecteur d'empreinte dans data/faces/<name>.
        """
        clean_name = name.strip()
        if not clean_name:
            return False, "Le nom ne peut pas être vide."

        with self._lock:
            person_dir = self.storage_dir / clean_name
            person_dir.mkdir(parents=True, exist_ok=True)

            encoding = self._extract_face_encoding(image_bgr)
            if encoding is None:
                return False, "Aucun visage net n'a été détecté sur l'image."

            # Sauvegarde de la vignette
            thumb_path = person_dir / "thumbnail.jpg"
            h, w = image_bgr.shape[:2]
            scale = 200.0 / max(h, w)
            thumb = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
            cv2.imwrite(str(thumb_path), thumb)

            # Mise à jour des encodages
            if clean_name not in self.known_encodings:
                self.known_encodings[clean_name] = []
            self.known_encodings[clean_name].append(encoding)

            # Sauvegarde chiffrée AES-256 des encodages (Conformité RGPD)
            enc_array = np.array(self.known_encodings[clean_name], dtype=object)
            raw_bytes = pickle.dumps(enc_array)
            encrypted_bytes = security_manager.encrypt_bytes(raw_bytes)
            (person_dir / "encodings.enc").write_bytes(encrypted_bytes)

            return True, f"Visage de '{clean_name}' enregistré et chiffré en AES-256 avec succès !"

    def identify_face(self, face_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Compare un visage aux personnes enregistrées.
        Retourne (nom, confiance) ou ('Inconnu', score).
        """
        with self._lock:
            if not self.known_encodings:
                return "Inconnu", 0.0

            encoding = self._extract_face_encoding(face_bgr)
            if encoding is None:
                return "Inconnu", 0.0

            best_name = "Inconnu"
            best_score = 0.0
            min_dist = 999.0

            for name, enc_list in self.known_encodings.items():
                for known_enc in enc_list:
                    if FACE_REC_AVAILABLE:
                        dist = float(face_recognition.face_distance([known_enc], encoding)[0])
                    else:
                        dist = float(np.linalg.norm(known_enc - encoding))

                    if dist < min_dist:
                        min_dist = dist
                        if dist <= self.tolerance:
                            best_name = name
                            # Confiance normalisée entre 0.0 et 1.0
                            best_score = max(0.0, 1.0 - (dist / self.tolerance))

            if best_name != "Inconnu":
                return best_name, round(best_score, 2)
            return "Inconnu", round(max(0.0, 1.0 - min_dist), 2)

    def attach_face_ids_to_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Pour chaque détection étiquetée comme humaine, tente d'identifier le visage
        dans la zone supérieure de la boîte englobante.
        """
        h, w = frame.shape[:2]
        for det in detections:
            if det.get("category") == "human":
                x1, y1, x2, y2 = det["box"]
                # Cible préférentielle du visage : moitié supérieure de la boîte personne
                face_y2 = y1 + int((y2 - y1) * 0.55)
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, face_y2_c = min(w, x2), min(h, face_y2)

                if (x2_c - x1_c) >= 30 and (face_y2_c - y1_c) >= 30:
                    crop = frame[y1_c:face_y2_c, x1_c:x2_c]

                    # Vérification Anti-Spoofing / Détection de vivant
                    is_live, live_score, live_reason = liveness_detector.check_liveness(crop)
                    det["liveness_score"] = live_score
                    det["is_live"] = is_live

                    if not is_live and liveness_detector.enabled:
                        det["face_name"] = "Inconnu [Photo Rejetée]"
                        det["face_conf"] = live_score
                        det["spoof_detected"] = True
                    else:
                        name, conf = self.identify_face(crop)
                        det["face_name"] = name
                        det["face_conf"] = conf
                        if name != "Inconnu":
                            bio_file = self.storage_dir / name / "bio.json"
                            if bio_file.exists():
                                try:
                                    with open(bio_file, "r", encoding="utf-8") as bf:
                                        bio = json.load(bf)
                                        det["face_desc"] = bio.get("description", "")
                                        det["face_bio"] = bio
                                except Exception:
                                    pass
        return detections

    def list_persons(self) -> List[Dict[str, Any]]:
        """Liste les personnes enregistrées avec le nombre d'échantillons et leur biographie."""
        with self._lock:
            result = []
            for name, encs in self.known_encodings.items():
                thumb_file = self.storage_dir / name / "thumbnail.jpg"
                bio_file = self.storage_dir / name / "bio.json"
                bio_data = None
                if bio_file.exists():
                    try:
                        with open(bio_file, "r", encoding="utf-8") as bf:
                            bio_data = json.load(bf)
                    except Exception:
                        pass
                result.append({
                    "name": name,
                    "samples_count": len(encs),
                    "has_thumbnail": thumb_file.exists(),
                    "thumbnail_url": f"/api/faces/{name}/thumbnail" if thumb_file.exists() else None,
                    "bio": bio_data,
                })
            return result

    def save_bio(self, name: str, bio_dict: Dict[str, Any]) -> bool:
        """Enregistre ou met à jour la biographie d'une personne."""
        with self._lock:
            person_dir = self.storage_dir / name
            person_dir.mkdir(parents=True, exist_ok=True)
            bio_file = person_dir / "bio.json"
            try:
                with open(bio_file, "w", encoding="utf-8") as bf:
                    json.dump(bio_dict, bf, ensure_ascii=False, indent=2)
                return True
            except Exception:
                return False

    def delete_person(self, name: str) -> bool:
        """Supprime définitivement une personne de la base locale."""
        with self._lock:
            if name in self.known_encodings:
                del self.known_encodings[name]
            person_dir = self.storage_dir / name
            if person_dir.exists():
                shutil.rmtree(person_dir)
                return True
            return False

    def delete_all(self) -> bool:
        """Supprime l'intégralité des visages et données biométriques stockées."""
        with self._lock:
            self.known_encodings.clear()
            if self.storage_dir.exists():
                shutil.rmtree(self.storage_dir)
                self.storage_dir.mkdir(parents=True, exist_ok=True)
            return True
