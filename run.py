"""
Script de lancement unique du projet de Vision par Ordinateur.
Usage :
    python run.py           -> HTTP port 8000 (PC local, sans avertissement SSL)
    python run.py --https   -> HTTPS port 8443 (accès smartphone sur le même réseau)
    python run.py --desktop -> Fenêtre OpenCV locale
"""

from pathlib import Path
import subprocess
import sys

# Auto-reconnexion au venv du projet si exécuté avec le Python global
BASE_DIR = Path(__file__).resolve().parent
venv_python = BASE_DIR / "venv" / "Scripts" / "python.exe"
if venv_python.exists():
    try:
        current_exe = Path(sys.executable).resolve()
        target_exe = venv_python.resolve()
        if current_exe != target_exe:
            sys.exit(subprocess.call([str(target_exe)] + sys.argv))
    except Exception:
        pass

import webbrowser

# Reconfiguration UTF-8 pour éviter les erreurs d'encodage sur console Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ajout du dossier racine au sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import cfg

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False


def get_local_ip() -> str:
    """Détecte l'adresse IP locale du PC sur le réseau Wi-Fi/LAN."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def print_banner(local_ip: str, port: int, use_https: bool) -> None:
    """Affiche une bannière d'information avec URLs et QR code."""
    scheme = "https" if use_https else "http"
    pc_url = f"{scheme}://localhost:{port}"
    phone_url = f"{scheme}://{local_ip}:{port}"

    print("\n" + "=" * 70)
    print("   [VISION IA TEMPS REEL] - DETECTION & RECONNAISSANCE FACIALE")
    print("   [OSINT] Recherche d'identite Web & Reseaux Sociaux")
    print("=" * 70)
    print(f"  * Acces PC local      :  {pc_url}")
    print(f"  * Acces Telephone     :  {phone_url}")
    print("=" * 70)
    if use_https:
        print("  [HTTPS] Certificat auto-signe actif (mode mobile).")
        print("  [INFO]  Si le navigateur bloque : tapez 'thisisunsafe' (sans Enter) sur Chrome")
        print("          ou Android Chrome : 'Parametres avances' -> 'Continuer vers le site'")
    else:
        print("  [HTTP]  Mode local sans SSL -> acces direct SANS avertissement.")
        print("  [INFO]  Pour acceder depuis un smartphone : relancez avec --https")
    print("-" * 70)
    print(f"  Ouvrez votre navigateur sur : {pc_url}\n")
    if QRCODE_AVAILABLE and use_https:
        try:
            qr = qrcode.QRCode(border=1)
            qr.add_data(phone_url)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        except Exception:
            pass
    print("=" * 70 + "\n")


def main():
    """Point d'entrée principal."""
    use_https = "--https" in sys.argv

    # Lancement du mode Desktop natif (OpenCV) si spécifié
    if "--desktop" in sys.argv or "desktop" in sys.argv:
        from desktop import run_desktop
        run_desktop()
        return

    local_ip = get_local_ip()

    if use_https:
        # Mode HTTPS : génération du certificat SSL (pour mobile)
        from app.certs import ensure_ssl_certs
        cert_path, key_path, local_ip = ensure_ssl_certs(BASE_DIR)
        port = int(cfg.get("server", "port", default=8443))
    else:
        # Mode HTTP par défaut : port 8000, aucun certificat requis
        cert_path, key_path = None, None
        port = 8000

    host = cfg.get("server", "host", default="0.0.0.0")

    # Affichage de la bannière
    print_banner(local_ip, port, use_https)

    # Ouverture automatique du navigateur PC
    scheme = "https" if use_https else "http"
    try:
        webbrowser.open(f"{scheme}://localhost:{port}")
    except Exception:
        pass

    # Démarrage du serveur Uvicorn
    import uvicorn
    uvicorn_kwargs: dict = dict(
        app="app.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
    if use_https and cert_path and key_path:
        uvicorn_kwargs["ssl_certfile"] = str(cert_path)
        uvicorn_kwargs["ssl_keyfile"] = str(key_path)

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
