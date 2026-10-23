# ==============================================================================
# Dockerfile Production - Vision IA Temps Réel (FastAPI & YOLOv8)
# ==============================================================================

FROM python:3.10-slim

# Empêche la génération de fichiers .pyc et active le flush immédiat des logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    ENVIRONMENT=production \
    REQUIRE_AUTH=true

WORKDIR /app

# Installation des dépendances système requises pour OpenCV, PyTorch et le traitement multimédia
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Création de l'arborescence de données persistantes
RUN mkdir -p data/faces data/captures/osint certs static

# Copie du code source applicatif
COPY app/ ./app/
COPY static/ ./static/
COPY config.yaml ./config.yaml
COPY yolov8n.pt ./yolov8n.pt

# Création d'un utilisateur non-root pour le principe de moindre privilège (Sécurité Conteneur)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Exposition des ports (8000 pour HTTP interne / proxy Nginx, 8443 pour SSL direct)
EXPOSE 8000 8443

# Sonde de santé Docker intégrée
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Lancement du serveur applicatif avec Uvicorn
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
