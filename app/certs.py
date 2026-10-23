"""
Module de gestion et de génération automatique de certificats TLS/SSL auto-signés.
Nécessaire car les navigateurs mobiles (iOS Safari, Android Chrome) exigent
strictement une connexion HTTPS pour autoriser l'accès à getUserMedia (caméra).
"""

import datetime
import ipaddress
from pathlib import Path
import socket
from typing import Tuple

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def get_local_ip() -> str:
    """Détecte l'adresse IP locale de la machine sur le réseau Wi-Fi/Ethernet."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Ne crée pas de connexion réelle, mais permet à l'OS de déterminer l'interface de sortie
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def generate_self_signed_cert(
    cert_path: Path,
    key_path: Path,
    valid_days: int = 365,
) -> Tuple[Path, Path]:
    """
    Génère un certificat X.509 auto-signé avec SAN (Subject Alternative Name)
    pour localhost, 127.0.0.1 et l'IP LAN de la machine.
    """
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    local_ip_str = get_local_ip()

    # Génération de la clé privée RSA 2048 bits
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )

    # Identité du certificat
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "FR"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "France"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "VisionAI"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Vision Temps Reel"),
        x509.NameAttribute(NameOID.COMMON_NAME, local_ip_str),
    ])

    # Noms alternatifs (SAN) : indispensables pour Chrome et iOS Safari
    san_list = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    try:
        san_list.append(x509.IPAddress(ipaddress.IPv4Address(local_ip_str)))
    except ValueError:
        pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=valid_days))
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .sign(private_key, hashes.SHA256(), default_backend())
    )

    # Sauvegarde de la clé privée
    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Sauvegarde du certificat
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    return cert_path, key_path


def ensure_ssl_certs(base_dir: Path) -> Tuple[Path, Path, str]:
    """Vérifie ou génère les certificats nécessaires et retourne (cert, key, local_ip)."""
    certs_dir = base_dir / "certs"
    cert_file = certs_dir / "cert.pem"
    key_file = certs_dir / "key.pem"
    local_ip = get_local_ip()

    if not cert_file.exists() or not key_file.exists():
        print(f"[SSL] Génération automatique du certificat TLS auto-signé pour {local_ip}...")
        generate_self_signed_cert(cert_file, key_file)
        print(f"[SSL] Certificat créé dans : {cert_file}")
    else:
        print(f"[SSL] Certificats existants trouvés : {cert_file}")

    return cert_file, key_file, local_ip
