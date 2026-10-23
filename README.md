# 🛡️ SentinAI - Vision IA Temps Réel & Sécurité Edge

> **SentinAI** est une plateforme open-source complète de vision par ordinateur et de sûreté intelligente : détection d'objets (YOLOv8 & YOLO-World), reconnaissance faciale locale chiffrée AES-256 (RGPD/Zero-Trust), analyse comportementale (zones restreintes, détection de chutes) et streaming multi-sources simultané (Webcam PC, Smartphone via QR code/HTTPS, Caméras IP industrielles RTSP/ONVIF).

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00599C.svg)](https://ultralytics.com/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## 📋 Table des Matières
1. [Architecture & Arborescence](#-architecture--arborescence)
2. [Installation Pas à Pas](#-installation-pas-à-pas)
3. [Lancement Rapide (Commande Unique)](#-lancement-rapide-commande-unique)
4. [Déploiement Production (Docker & Nginx)](#-déploiement-production-docker--nginx)
5. [Connexion du Téléphone & HTTPS](#-connexion-du-téléphone--https)
6. [Guide des Fonctionnalités](#-guide-des-fonctionnalités)
7. [Dépannage & FAQ](#-dépannage--faq)

---

## 📁 Architecture & Arborescence

```text
SentinAI/

├── config.yaml               # Configuration générale (seuils, modèles, caméras, cloud)
├── requirements.txt          # Dépendances Python
├── run.py                    # Script de lancement unique (génération SSL, QR code, Uvicorn)
├── README.md                 # Guide complet d'utilisation et dépannage
├── app/
│   ├── __init__.py           # Package app
│   ├── config.py             # Gestionnaire thread-safe de la configuration
│   ├── certs.py              # Générateur de certificat SSL auto-signé (SubjectAltName)
│   ├── detector.py           # Inférence YOLOv8 & YOLO-World (CUDA/MPS/CPU + annotation)
│   ├── faces.py              # Reconnaissance faciale & gestion locale des identités
│   ├── cloud_fallback.py     # Secours Google Cloud Vision / Azure Vision
│   ├── camera_stream.py      # Acquisition webcam PC OpenCV sans latence
│   └── server.py             # Serveur FastAPI, WebSockets & routes REST
├── certs/                    # Dossier auto-généré contenant cert.pem et key.pem
├── data/
│   ├── faces/                # Empreintes faciales vectorielles et miniatures locales
│   └── captures/             # Clichés et instantanés annotés
└── static/
    ├── index.html            # Interface web responsive PC & Mobile
    ├── css/
    │   └── style.css         # Thème sombre moderne, glassmorphism & HUD
    └── js/
        └── app.js            # Streaming WebSocket, getUserMedia, switch caméras
```

---

## ⚙️ Installation Pas à Pas

### Prérequis
- Python 3.10 ou supérieur installé.
- Une webcam sur le PC ou un smartphone connecté au **même réseau Wi-Fi**.

### 1. Cloner ou ouvrir le dossier du projet
```bash
cd "voir-tout"
```

### 2. Créer un environnement virtuel (recommandé)

- **Sur Windows (PowerShell) :**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

- **Sur macOS / Linux :**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

> **Note GPU (NVIDIA CUDA) :** Pour profiter de l'accélération GPU maximale sur Windows/Linux, vous pouvez installer la version PyTorch compatible CUDA via :
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> *Sur macOS, l'accélération matérielle Apple Silicon (MPS) est activée nativement sans configuration additionnelle.*

---

## 🚀 Lancement Rapide (Commande Unique)

Lancez simplement la commande :

```bash
python run.py
```

### Que fait cette commande ?
1. **Détecte automatiquement** l'adresse IP locale de votre machine (ex: `192.168.1.42`).
2. **Génère un certificat SSL auto-signé** (nécessaire pour activer la caméra sur mobile).
3. **Affiche dans le terminal un QR Code ASCII** scannable directement avec votre téléphone.
4. **Ouvre automatiquement** l'interface web dans votre navigateur PC (`https://localhost:8443`).
5. **Démarre le serveur FastAPI** en HTTPS sur le port 8443.

---

## 📱 Connexion du Téléphone & HTTPS

Les navigateurs modernes (iOS Safari, Android Chrome) imposent une connexion **HTTPS sécurisée** pour autoriser l'accès aux caméras via `navigator.mediaDevices.getUserMedia`.

### Procédure de première connexion :
1. Connectez votre téléphone au **même réseau Wi-Fi** que votre PC.
2. Scannez le **QR Code** affiché dans votre terminal (ou tapez l'adresse affichée, ex: `https://192.168.1.42:8443`).
3. Le navigateur affichera un avertissement de sécurité standard (car le certificat a été généré localement pour votre propre réseau) :
   - **Sur Android (Google Chrome) :** Cliquez sur **"Paramètres avancés"** puis sur **"Continuer vers le site (non sécurisé)"**.
   - **Sur iPhone (Apple Safari) :** Cliquez sur **"Afficher les détails"** puis sur **"Visiter ce site web"** et confirmez.
4. Autorisez l'accès à la caméra lorsque le navigateur vous le demande.
5. **C'est prêt !** Le flux de votre téléphone s'affiche en direct, annoté avec les cadres de détection.

---

## 🎯 Guide des Fonctionnalités

### 1. Mode Détection Standard (YOLOv8)
- Par défaut, le modèle ultraléger `yolov8n.pt` est utilisé pour offrir une fluidité maximale (> 30 FPS sur GPU, > 15-20 FPS sur CPU moyen).
- Dans l'onglet **"🧠 Modèle IA"**, vous pouvez basculer en un clic vers `yolov8s.pt` ou `yolov8m.pt` pour une meilleure précision.
- Ajustez le curseur **"Seuil de Confiance"** pour filtrer les détections incertaines.

### 2. Mode Vocabulaire Ouvert "Tout Reconnaître" (YOLO-World)
- Ouvrez l'onglet **"🌍 Mode Ouvert"** et activez le switch.
- Tapez n'importe quelle liste d'objets en texte libre (ex: `sac à dos bleu, tasse de café, montre, clé, chat`).
- Cliquez sur **"Mettre à jour les classes YOLO-World"** : le modèle détecte ces objets spécifiques immédiatement, **sans aucun réentraînement** !

### 3. Reconnaissance Faciale Locale & RGPD
- **Stockage 100% local :** Aucune image ni empreinte biométrique n'est envoyée vers des serveurs tiers. Tout reste stocké dans le dossier local `data/faces/`.
- **Enregistrer une personne :**
  1. Allez dans l'onglet **"👤 Visages"**.
  2. Entrez le nom ou prénom de la personne.
  3. Cliquez sur **"📸 Capturer depuis le flux actuel"** (ou téléversez une photo existante).
  4. La personne est immédiatement mémorisée et sera reconnue en temps réel avec son nom.
  5. Les personnes non enregistrées affichent le badge **"Inconnu"**.
- **Droit à l'oubli (RGPD) :** Vous pouvez supprimer individuellement une personne via son bouton corbeille 🗑️, ou cliquer sur **"🗑️ Tout effacer (RGPD)"** pour purger intégralement la base.

### 4. Captures & Statistiques à l'Écran
- **HUD dynamique :** Compteur de FPS réels, latence de traitement en millisecondes et nombre d'objets visibles.
- **Bouton photo 📸 :** Enregistre instantanément la trame vidéo avec ses boîtes de détection dans `data/captures/`.
- **Galerie :** Consultez et téléchargez vos captures dans l'onglet **"🖼️ Captures"**.
- **Bascule caméra 🔄 :** Permet sur smartphone de passer instantanément de la caméra arrière (pour analyser l'environnement) à la caméra avant (pour les visages).

### 5. Option de Secours Cloud
Dans `config.yaml`, vous pouvez activer `cloud_fallback: enabled: true`. Si le modèle local ne détecte aucun objet avec une confiance suffisante, l'image est alors soumise en arrière-plan à l'API Google Cloud Vision ou Azure AI Vision.

---

## 🛠️ Dépannage & FAQ

### 1. La page ne charge pas sur le téléphone
- Vérifiez que le PC et le smartphone sont connectés à la **même box / point d'accès Wi-Fi** (attention aux réseaux "Invité" qui isolent les appareils).
- Vérifiez le pare-feu de votre PC :
  - **Sur Windows :** Autorisez Python dans le Pare-feu Windows Defender, ou autorisez temporairement le port 8443 :
    ```powershell
    New-NetFirewallRule -DisplayName "VisionAI" -Direction Inbound -LocalPort 8443 -Protocol TCP -Action Allow
    ```
  - **Sur Linux :** `sudo ufw allow 8443/tcp`

### 2. La caméra ne s'active pas sur le téléphone
- Avez-vous bien accédé à l'URL en **`https://`** et non `http://` ? Sans HTTPS, les navigateurs mobiles bloquent l'API `getUserMedia`.
- Vérifiez dans les paramètres de votre navigateur mobile que l'autorisation caméra est accordée au site.

### 3. La webcam PC OpenCV affiche un écran noir ou une erreur
- Si vous avez plusieurs caméras (webcam interne + webcam USB ou DroidCam/OBS Virtual Cam), changez l'index de caméra dans l'onglet **"📹 Source"** (index `0`, `1` ou `2`).
- Vérifiez qu'aucune autre application (Teams, Zoom, Discord) n'utilise la caméra en même temps.

### 4. Le débit d'images est lent
- Assurez-vous d'utiliser `yolov8n.pt` (modèle nano).
- Dans `config.yaml`, vous pouvez réduire la taille d'image d'inférence à `img_size: 480`.
- Si vous possédez une carte graphique NVIDIA, vérifiez que PyTorch CUDA est bien actif (la pastille en haut à droite indiquera `GPU (CUDA)`).

---

## 🚀 Pistes d'Amélioration & Déploiement Embarqué

1. **Export TensorRT / ONNX Runtime :**
   Exportez le modèle YOLO pour diviser par 2 la latence d'inférence sur GPU NVIDIA :
   ```bash
   yolo export model=yolov8n.pt format=engine device=0
   ```
2. **Déploiement sur Raspberry Pi 4 / 5 :**
   - Utilisez le format **NCNN** ou **TFLite** (`yolo export model=yolov8n.pt format=ncnn`) pour atteindre 20+ FPS sur le processeur ARM du Raspberry Pi sans carte graphique.
3. **Application Mobile Autonome (PWA / Capacitor) :**
   - L'interface web est déjà responsive et conforme aux standards PWA. Vous pouvez ajouter un fichier `manifest.json` pour permettre l'installation en "Application Web Progressive" sur l'écran d'accueil iOS/Android d'un simple geste.
