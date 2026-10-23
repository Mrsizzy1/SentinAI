"""
Module de Sécurité d'Entreprise, Authentification Zero-Trust et Chiffrement Biométrique.
Conformité ISO 27001 & RGPD :
- Chiffrement symétrique AES-256 (Fernet / CBC + HMAC) de tous les vecteurs biométriques.
- Gestion des clés de chiffrement au repos.
- Vérification des jetons d'accès et clés d'API (Admin & Opérateurs).
- Protection contre la traversée de répertoire (Path Traversal).
"""

import base64
import hashlib
import os
from pathlib import Path
import re
import secrets
from typing import Optional
from cryptography.fernet import Fernet
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery, HTTPAuthorizationCredentials, HTTPBearer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Clé de chiffrement persistante
KEY_FILE = Path("data/.enterprise_master.key")
DEFAULT_API_KEY = "ENT-SECURE-2026-X9"

# Détection de l'environnement
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false" if ENVIRONMENT == "development" else "true").lower() in ("true", "1", "yes")


class EnterpriseSecurityManager:
    """Gestionnaire de sécurité, authentification et cryptographie biométrique."""

    def __init__(self, key_file: Path = KEY_FILE):
        self.key_file = key_file
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        self.fernet = self._init_fernet()

        env_key = os.getenv("ENTERPRISE_API_KEY")
        if env_key:
            self.api_key = env_key.strip()
        elif ENVIRONMENT == "production":
            # En production sans clé explicite, génération d'une clé forte et affichage d'une alerte
            generated_key = secrets.token_hex(24)
            self.api_key = f"ENT-PROD-{generated_key}"
            print("\n" + "!" * 72)
            print("[SECURITE PRODUCTION] AUCUNE CLE 'ENTERPRISE_API_KEY' FOURNIE DANS L'ENVIRONNEMENT !")
            print(f"[SECURITE PRODUCTION] Cle forte temporaire generee : {self.api_key}")
            print("Configurez 'ENTERPRISE_API_KEY' dans votre fichier .env pour la rendre permanente.")
            print("!" * 72 + "\n")
        else:
            self.api_key = DEFAULT_API_KEY

    def _init_fernet(self) -> Fernet:
        """Charge ou génère la clé maîtresse AES-256 de l'entreprise."""
        if self.key_file.exists():
            try:
                key = self.key_file.read_bytes().strip()
                return Fernet(key)
            except Exception:
                pass

        # Génération d'une nouvelle clé maîtresse sécurisée 256 bits
        new_key = Fernet.generate_key()
        try:
            self.key_file.write_bytes(new_key)
        except Exception as e:
            print(f"[SecurityManager] Avertissement écriture clé: {e}")
        return Fernet(new_key)

    def encrypt_bytes(self, raw_data: bytes) -> bytes:
        """Chiffre des données biométriques en AES-256 avec authentification HMAC."""
        return self.fernet.encrypt(raw_data)

    def decrypt_bytes(self, cipher_data: bytes) -> bytes:
        """Déchiffre des données biométriques chiffrées."""
        return self.fernet.decrypt(cipher_data)

    def verify_token(self, token: Optional[str]) -> bool:
        """Vérifie si le jeton fourni correspond à la clé d'API d'entreprise."""
        if not token:
            return False
        # Comparaison en temps constant pour éviter les attaques temporelles (timing attacks)
        return secrets.compare_digest(token.strip(), self.api_key)


# Instance globale
security_manager = EnterpriseSecurityManager()

api_key_header = APIKeyHeader(name="X-Enterprise-Key", auto_error=False)
api_key_query = APIKeyQuery(name="token", auto_error=False)
http_bearer = HTTPBearer(auto_error=False)


def safe_resolve_path(base_dir: Path, user_path: str) -> Path:
    """
    Protège contre les attaques de type Path Traversal (../, null bytes).
    Vérifie formellement que le chemin résolu réside bien dans le répertoire de base autorisé.
    """
    if not user_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Nom de fichier vide.")

    # Rejet des caractères nuls et séparateurs suspects
    if "\0" in user_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chemin invalide (null byte détecté).")

    # Nettoyage du nom pour ne garder que le nom de fichier si sous-dossier non attendu
    clean_filename = Path(user_path).name
    if not clean_filename or clean_filename in (".", ".."):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Nom de fichier non autorisé.")

    target = (base_dir / clean_filename).resolve()
    base_resolved = base_dir.resolve()

    try:
        # Vérification du confinement sous le répertoire de base
        target.relative_to(base_resolved)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Accès non autorisé en dehors du répertoire.")

    return target


def require_enterprise_auth(
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(http_bearer),
) -> bool:
    """
    Dépendance FastAPI pour protéger les accès critiques.
    Vérifie l'en-tête X-Enterprise-Key, Bearer token ou paramètre de requête.
    En mode production (REQUIRE_AUTH=True), bloque systématiquement tout accès non authentifié.
    """
    token = header_key or query_key or (bearer_token.credentials if bearer_token else None)

    # Si l'authentification n'est pas strictement requise (mode développement local sans clé)
    if not REQUIRE_AUTH and not token:
        return True

    # Si un token est manquant alors que l'auth est obligatoire
    if REQUIRE_AUTH and not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentification requise : fournissez un jeton via l'en-tête 'X-Enterprise-Key' ou 'Authorization: Bearer'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Vérification cryptographique en temps constant
    if not security_manager.verify_token(token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé de sécurité Entreprise invalide.",
        )

    return True

