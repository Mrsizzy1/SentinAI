"""
Module d'Intelligence & Reconnaissance d'Identité Web (Wikipedia, Wikidata, Reverse Search).
Permet au système :
1. D'enrichir l'identification des personnes avec leur biographie et profession (Wikipédia).
2. D'apprendre automatiquement de nouvelles célébrités / personnalités depuis le Web en 2 secondes.
3. De lancer une recherche inversée d'identité (Google Lens) sur n'importe quel visage inconnu.
"""

from datetime import datetime
import json
from pathlib import Path
import re
import threading
from typing import Any, Dict, Optional, Tuple
import urllib.parse
import webbrowser
import cv2
import numpy as np
import requests


class WebIdentityEngine:
    """Moteur de recherche et d'enrichissement d'identité en ligne."""

    def __init__(self, cache_file: str = "data/web_cache.json"):
        self.cache_path = Path(cache_file)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Charge le cache local d'informations Wikipédia pour éviter les requêtes répétées."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self) -> None:
        """Sauvegarde le cache sur disque."""
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _search_wikipedia_title(self, query: str, lang: str = "fr") -> Optional[str]:
        """Effectue une recherche plein texte pour trouver le titre exact de la page Wikipédia."""
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        }
        headers = {"User-Agent": "VisionAI-Live/2.0 (computer-vision-project@local.dev)"}
        try:
            res = requests.get(search_url, params=params, headers=headers, timeout=3.0)
            if res.status_code == 200:
                data = res.json()
                results = data.get("query", {}).get("search", [])
                if results:
                    return results[0].get("title")
        except Exception:
            pass
        return None

    def get_person_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations biographiques (profession, description, résumé)
        d'une personne via l'API Wikipédia (FR puis EN, avec recherche automatique de titre).
        """
        clean_name = name.strip()
        if not clean_name or clean_name.lower() in ("inconnu", "unknown", "humain"):
            return None

        # Vérification en cache mémoire
        with self._lock:
            if clean_name.lower() in self.cache:
                return self.cache[clean_name.lower()]

        headers = {
            "User-Agent": "VisionAI-Live/2.0 (computer-vision-project@local.dev)"
        }

        # 1. Tentative avec le nom direct (FR)
        languages = ["fr", "en"]
        for lang in languages:
            target_title = clean_name
            encoded_title = urllib.parse.quote(target_title.replace(" ", "_"))
            summary_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"

            try:
                res = requests.get(summary_url, headers=headers, timeout=3.0)
                # Si 404, tenter la recherche plein texte pour trouver la bonne page
                if res.status_code == 404:
                    found_title = self._search_wikipedia_title(clean_name, lang=lang)
                    if found_title:
                        encoded_title = urllib.parse.quote(found_title.replace(" ", "_"))
                        summary_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
                        res = requests.get(summary_url, headers=headers, timeout=3.0)

                if res.status_code == 200:
                    data = res.json()
                    desc = data.get("description", "Personnalité publique" if lang == "fr" else "Public figure")
                    extract = data.get("extract", "")
                    if len(extract) > 220:
                        extract = extract[:220].rsplit(" ", 1)[0] + "..."

                    info = {
                        "name": data.get("title", clean_name),
                        "description": desc,
                        "extract": extract,
                        "url": data.get("content_urls", {}).get("desktop", {}).get("page", f"https://{lang}.wikipedia.org/wiki/{encoded_title}"),
                        "image_url": data.get("thumbnail", {}).get("source"),
                        "source": f"Wikipédia {lang.upper()}",
                    }
                    with self._lock:
                        self.cache[clean_name.lower()] = info
                        self.cache[info["name"].lower()] = info
                        self._save_cache()
                    return info
            except Exception:
                pass

        return None

    def learn_person_from_web(self, person_name: str, face_engine: Any) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Télécharge automatiquement la photo officielle d'une personne depuis Wikipédia,
        extrait son visage et l'ajoute directement à la base de reconnaissance locale.
        """
        clean_name = person_name.strip()
        if not clean_name:
            return False, "Veuillez saisir un nom valide.", None

        info = self.get_person_info(clean_name)
        if not info or not info.get("image_url"):
            return False, f"Impossible de trouver une photo officielle pour '{clean_name}' sur Wikipédia.", None

        image_url = info["image_url"]
        # Récupération de l'image haute résolution si miniature
        high_res_url = re.sub(r"/\d+px-", "/640px-", image_url) if "/thumb/" in image_url else image_url
        headers = {"User-Agent": "VisionAI-Live/2.0 (computer-vision-project@local.dev)"}

        try:
            img_res = requests.get(high_res_url, headers=headers, timeout=6.0)
            if img_res.status_code != 200:
                # Repli sur l'URL originale
                img_res = requests.get(image_url, headers=headers, timeout=6.0)

            if img_res.status_code != 200:
                return False, f"Échec de téléchargement de l'image ({img_res.status_code}).", None

            # Décodage OpenCV de l'image
            img_array = np.frombuffer(img_res.content, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return False, "Format d'image non reconnu.", None

            registered_name = info["name"]
            success, msg = face_engine.register_person(registered_name, img_bgr)
            if success:
                # Sauvegarde des métadonnées biographiques
                storage_root = getattr(face_engine, "storage_dir", Path("data/faces"))
                person_dir = Path(storage_root) / registered_name
                person_dir.mkdir(parents=True, exist_ok=True)
                with open(person_dir / "bio.json", "w", encoding="utf-8") as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)

                desc = info.get("description", "")
                return True, f"'{registered_name}' ({desc}) appris avec succès depuis Wikipédia !", info
            else:
                return False, f"Échec extraction faciale : {msg}", None

        except Exception as e:
            return False, f"Erreur lors de l'apprentissage web : {e}", None

    def search_face_online(self, face_bgr: np.ndarray, open_browser: bool = True) -> str:
        """
        Sauvegarde le visage inconnu dans data/captures et ouvre Google Lens
        pour identifier immédiatement où cette personne apparaît sur le Web.
        """
        captures_dir = Path("data/captures")
        captures_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_path = captures_dir / f"recherche_visage_{ts}.jpg"
        latest_path = captures_dir / "derniere_recherche_visage.jpg"

        cv2.imwrite(str(face_path), face_bgr)
        cv2.imwrite(str(latest_path), face_bgr)

        # Lien vers Google Lens
        lens_url = "https://lens.google.com/"
        if open_browser:
            try:
                webbrowser.open(lens_url)
            except Exception:
                pass

        return str(face_path)


# Instance singleton globale
web_identity = WebIdentityEngine()
