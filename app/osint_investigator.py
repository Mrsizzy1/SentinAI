"""
Module d'Investigation OSINT & Recherche Inverse Multimoteur (Webcam -> Réseaux Sociaux & Identité).
Permet au système :
1. D'isoler n'importe quel visage inconnu capturé par la webcam.
2. De générer les requêtes de recherche inversée pour les 5 plus grands moteurs visuels (Google Lens, Yandex, Bing, PimEyes, FaceCheck).
3. D'exécuter un profiling automatisé sur les réseaux sociaux (LinkedIn, Instagram, Facebook, X, GitHub) dès qu'un nom est identifié.
4. De créer un Dossier d'Investigation complet et de l'enrôler pour que la caméra l'affiche désormais avec son nom et ses profils.
"""

from datetime import datetime
import json
from pathlib import Path
import urllib.parse
import webbrowser
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import requests


class OSINTInvestigator:
    """Moteur d'enquête et de recherche d'identité en ligne multi-sources."""

    def __init__(self, osint_dir: str = "data/captures/osint"):
        self.osint_dir = Path(osint_dir)
        self.osint_dir.mkdir(parents=True, exist_ok=True)

    def prepare_target_investigation(self, face_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Découpe, normalise et prépare un visage inconnu pour l'investigation sur le Web.
        Génère les passerelles de recherche inversée et les dossiers de preuve.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_filename = f"cible_osint_{ts}.jpg"
        target_path = self.osint_dir / target_filename
        latest_path = self.osint_dir / "latest_osint_target.jpg"

        # Optimisation du contraste pour améliorer la reconnaissance visuelle
        enhanced = cv2.detailEnhance(face_bgr, sigma_s=10, sigma_r=0.15) if face_bgr.shape[0] >= 60 else face_bgr
        cv2.imwrite(str(target_path), enhanced)
        cv2.imwrite(str(latest_path), enhanced)

        # Moteurs de recherche inversée mondiaux
        search_engines = {
            "google_lens": {
                "name": "Google Lens (Recherche Visuelle Globale)",
                "url": "https://lens.google.com/",
                "description": "Indexe les photos de profils, articles de presse et blogs.",
            },
            "yandex_images": {
                "name": "Yandex Visual Search (Le plus puissant pour les visages)",
                "url": "https://yandex.com/images/search?rpt=imageview",
                "description": "Reconnu mondialement pour retrouver des visages même sous un autre angle.",
            },
            "bing_visual": {
                "name": "Bing Visual Search",
                "url": "https://www.bing.com/visualsearch",
                "description": "Moteur visuel Microsoft pour l'identification de personnalités.",
            },
            "facecheck_id": {
                "name": "FaceCheck.ID (Réseaux Sociaux & Web Ouvert)",
                "url": "https://facecheck.id/",
                "description": "Moteur spécialisé dans la détection de profils réseaux sociaux par photo.",
            },
        }

        return {
            "status": "success",
            "timestamp": ts,
            "image_filename": target_filename,
            "image_url": f"/api/captures/osint/{target_filename}",
            "file_path": str(target_path),
            "search_engines": search_engines,
        }

    def generate_social_dossier(self, person_name: str) -> Dict[str, Any]:
        """
        Génère un dossier d'investigation complet avec les liens de recherche directs
        vers les réseaux sociaux (LinkedIn, Instagram, Facebook, X, GitHub) et Wikipédia.
        """
        clean_name = person_name.strip()
        encoded = urllib.parse.quote(f'"{clean_name}"')

        # Liens directs de recherche profil
        social_links = {
            "linkedin": {
                "platform": "LinkedIn (Professionnel)",
                "search_url": f"https://www.google.com/search?q=site:linkedin.com/in/+{encoded}",
                "icon": "💼",
            },
            "instagram": {
                "platform": "Instagram",
                "search_url": f"https://www.google.com/search?q=site:instagram.com/+{encoded}",
                "icon": "📸",
            },
            "facebook": {
                "platform": "Facebook",
                "search_url": f"https://www.google.com/search?q=site:facebook.com/+{encoded}",
                "icon": "👥",
            },
            "twitter_x": {
                "platform": "X / Twitter",
                "search_url": f"https://www.google.com/search?q=site:x.com/+{encoded}",
                "icon": "🐦",
            },
            "github": {
                "platform": "GitHub (Développeur)",
                "search_url": f"https://www.google.com/search?q=site:github.com/+{encoded}",
                "icon": "💻",
            },
            "google_news": {
                "platform": "Actualités & Presse",
                "search_url": f"https://news.google.com/search?q={encoded}",
                "icon": "📰",
            },
        }

        # Tentative d'enrichissement automatique Wikipédia si personnalité
        wiki_bio = None
        try:
            from app.web_identity import web_identity
            wiki_bio = web_identity.get_person_info(clean_name)
        except Exception:
            pass

        return {
            "name": clean_name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "social_platforms": social_links,
            "wiki_bio": wiki_bio,
        }

    def register_investigated_profile(
        self,
        name: str,
        role: str,
        notes: str,
        image_path: str,
        face_engine: Any,
    ) -> Tuple[bool, str]:
        """
        Associe officiellement le nom et les informations trouvées en ligne au visage capturé,
        et l'enregistre immédiatement dans la base pour que la webcam le reconnaisse en direct !
        """
        clean_name = name.strip()
        if not clean_name:
            return False, "Le nom ne peut pas être vide."

        p = Path(image_path)
        if not p.exists():
            return False, f"Fichier image source introuvable: {image_path}"

        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            return False, "Échec de lecture de l'image cible."

        # Enregistrement biométrique
        success, msg = face_engine.register_person(clean_name, img_bgr)
        if not success:
            return False, msg

        # Sauvegarde de la fiche profil enrichie
        person_dir = Path(face_engine.storage_dir) / clean_name
        person_dir.mkdir(parents=True, exist_ok=True)

        dossier = self.generate_social_dossier(clean_name)
        dossier["role"] = role or "Identifié via Enquête Web"
        dossier["notes"] = notes or ""
        dossier["description"] = role or "Profil Web vérifié"

        with open(person_dir / "bio.json", "w", encoding="utf-8") as f:
            json.dump(dossier, f, ensure_ascii=False, indent=2)

        return True, f"Identité '{clean_name}' enregistrée avec succès ! La caméra le reconnaîtra désormais automatiquement."


# Instance globale
osint_investigator = OSINTInvestigator()
