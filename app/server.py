"""
Serveur FastAPI & WebSockets principal.
Gère :
- Le streaming bidirectionnel WebSocket pour la caméra du téléphone (getUserMedia)
- Le streaming WebSocket pour la webcam locale du PC (OpenCV)
- Les routes REST pour l'enregistrement/suppression des visages, la configuration dynamique
- L'enregistrement et consultation des captures d'écran
- Le service des fichiers statiques de l'application web responsive
"""

import asyncio
import base64
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from app.actions import action_dispatcher
from app.audit import audit_logger
from app.behavior import behavior_engine
from app.camera_stream import PCCameraStream
from app.cloud_fallback import CloudVisionFallback
from app.config import cfg
from app.detector import ObjectDetector
from app.faces import FaceEngine, GDPR_NOTICE
from app.liveness import liveness_detector
from app.osint_investigator import osint_investigator
from app.security import (
    ENVIRONMENT,
    REQUIRE_AUTH,
    require_enterprise_auth,
    safe_resolve_path,
    security_manager,
)
from app.web_identity import web_identity

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Vision IA Temps Réel Multi-Sources",
    description="Détection d'objets, personnes, animaux et reconnaissance faciale (PC & Mobile)",
    version="1.0.0",
)

# Configuration CORS sécurisée (paramétrable via variable d'environnement)
cors_env = os.getenv("CORS_ORIGINS", "*")
allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
CAPTURES_DIR = BASE_DIR / cfg.get("storage", "captures_dir", default="data/captures")
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# Moteurs IA
detector = ObjectDetector(
    model_name=cfg.get("yolo", "model_name", default="yolov8n.pt"),
    conf_threshold=float(cfg.get("yolo", "confidence_threshold", default=0.45)),
    iou_threshold=float(cfg.get("yolo", "iou_threshold", default=0.45)),
    img_size=int(cfg.get("yolo", "img_size", default=640)),
    open_vocab_enabled=bool(cfg.get("yolo", "open_vocabulary_enabled", default=False)),
    open_vocab_model=cfg.get("yolo", "open_vocabulary_model", default="yolov8s-world.pt"),
    custom_classes=cfg.get("yolo", "custom_classes", default=[]),
)

face_engine = FaceEngine(
    storage_dir=str(BASE_DIR / cfg.get("faces", "storage_dir", default="data/faces")),
    tolerance=float(cfg.get("faces", "tolerance", default=0.52)),
)

cloud_fallback = CloudVisionFallback(
    enabled=bool(cfg.get("cloud_fallback", "enabled", default=False)),
    provider=cfg.get("cloud_fallback", "provider", default="google"),
    api_key_or_creds=cfg.get("cloud_fallback", "api_key_or_creds", default=""),
    trigger_confidence_threshold=float(cfg.get("cloud_fallback", "trigger_confidence_threshold", default=0.35)),
)

pc_camera = PCCameraStream(
    camera_index=int(cfg.get("camera", "pc_camera_index", default=0)),
    width=int(cfg.get("camera", "capture_width", default=640)),
    height=int(cfg.get("camera", "capture_height", default=480)),
)


# Traitement commun d'une image (détection YOLO + visages + sûreté entreprise + annotation)
def process_single_frame(frame_bgr: np.ndarray, calc_fps: float) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, int], List[Dict[str, Any]], Dict[str, Any]]:
    # 1. Détection YOLO
    detections, _ = detector.detect(frame_bgr)

    # 2. Secours Cloud si aucun objet détecté ou confiance trop basse
    if cloud_fallback.is_active() and (len(detections) == 0 or max([d["confidence"] for d in detections], default=0.0) < cloud_fallback.trigger_threshold):
        cloud_dets = cloud_fallback.analyze_fallback(frame_bgr)
        if cloud_dets:
            detections.extend(cloud_dets)

    # 3. Reconnaissance faciale si activée
    if cfg.get("faces", "enabled", default=True):
        detections = face_engine.attach_face_ids_to_detections(frame_bgr, detections)

    # 4. Analyse Comportementale Entreprise (Zone interdite, chutes, errance, mouvements)
    active_alerts, behavior_summary = behavior_engine.analyze(frame_bgr, detections, audit_logger)

    # 5. Statistiques des catégories
    categories_count: Dict[str, int] = {}
    for d in detections:
        cat = d.get("category", "object")
        categories_count[cat] = categories_count.get(cat, 0) + 1

    # 6. Annotation graphique (boîtes + zone restreinte + bandeau alerte)
    annotated = detector.annotate_frame(
        frame_bgr,
        detections,
        fps=calc_fps,
        behavior_summary=behavior_summary,
        active_alerts=active_alerts,
    )
    return annotated, detections, categories_count, active_alerts, behavior_summary


# ==============================================================================
# WEBSOCKETS : Flux Vidéo Temps Réel
# ==============================================================================

@app.websocket("/ws/phone_stream")
async def websocket_phone_stream(websocket: WebSocket):
    """
    WebSocket pour la caméra du téléphone (getUserMedia) :
    Le smartphone envoie ses images capturées en JPEG base64,
    le serveur applique l'IA et renvoie l'image annotée avec les métadonnées.
    """
    await websocket.accept()
    last_frame_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            # Réception du payload JSON contenant la frame base64
            data_text = await websocket.receive_text()
            now = time.perf_counter()
            dt = now - last_frame_time
            if dt > 0:
                fps = 0.8 * fps + 0.2 * (1.0 / dt) if fps > 0 else 1.0 / dt
            last_frame_time = now

            try:
                data = json.loads(data_text)
                img_data_url = data.get("image", "")
            except Exception:
                img_data_url = data_text

            # Extraction des octets JPEG depuis le data URL
            if "," in img_data_url:
                _, b64_part = img_data_url.split(",", 1)
            else:
                b64_part = img_data_url

            img_bytes = base64.b64decode(b64_part)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text(json.dumps({"error": "Décodage impossible"}))
                continue

            # Traitement IA
            loop = asyncio.get_event_loop()
            annotated, detections, cat_counts, alerts, b_summary = await loop.run_in_executor(
                None, process_single_frame, frame, fps
            )

            # Encodage de l'image annotée
            _, enc_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(enc_buf).decode("utf-8")

            # Réponse envoyée au téléphone
            response_payload = {
                "image": annotated_b64,
                "fps": round(fps, 1),
                "latency_ms": round(detector.last_inference_time_ms, 1),
                "detections_count": len(detections),
                "detections": detections,
                "categories": cat_counts,
                "alerts": alerts,
                "behavior": b_summary,
                "open_vocab": detector.open_vocab_enabled,
            }
            await websocket.send_text(json.dumps(response_payload))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Phone] Déconnexion ou erreur: {e}")


@app.websocket("/ws/pc_stream")
async def websocket_pc_stream(websocket: WebSocket):
    """
    WebSocket pour la webcam locale du PC (OpenCV) :
    Le serveur lit sa caméra locale en continu et pousse les images annotées au navigateur.
    """
    await websocket.accept()
    if not pc_camera.is_opened:
        pc_camera.start()

    last_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            # Vérifie si le client demande une pause ou des commandes
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                # Client message handled if any
            except asyncio.TimeoutError:
                pass

            ret, frame = pc_camera.read()
            if not ret or frame is None:
                await asyncio.sleep(0.03)
                continue

            now = time.perf_counter()
            dt = now - last_time
            if dt > 0:
                fps = 0.8 * fps + 0.2 * (1.0 / dt) if fps > 0 else 1.0 / dt
            last_time = now

            # Inférence asynchrone pour ne pas bloquer la boucle d'événements
            loop = asyncio.get_event_loop()
            annotated, detections, cat_counts, alerts, b_summary = await loop.run_in_executor(
                None, process_single_frame, frame, fps
            )

            _, enc_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(enc_buf).decode("utf-8")

            payload = {
                "image": annotated_b64,
                "fps": round(fps, 1),
                "latency_ms": round(detector.last_inference_time_ms, 1),
                "detections_count": len(detections),
                "detections": detections,
                "categories": cat_counts,
                "alerts": alerts,
                "behavior": b_summary,
                "open_vocab": detector.open_vocab_enabled,
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket PC] Déconnexion: {e}")


# ==============================================================================
# ROUTES REST : Surveillance de Santé & Diagnostic (Production Healthchecks)
# ==============================================================================

@app.get("/health")
@app.get("/api/health")
async def health_check():
    """
    Vérification de l'état de santé du service pour load-balancer et orchestrateurs.
    Accessible sans authentification pour les sondes Kubernetes/Docker.
    """
    return {
        "status": "healthy",
        "environment": ENVIRONMENT,
        "auth_required": REQUIRE_AUTH,
        "timestamp": datetime.now().isoformat(),
        "device": detector.device,
        "active_model": detector._active_model_name,
        "camera_backend": pc_camera.backend_name,
        "known_faces_count": len(face_engine.known_encodings),
    }


# ==============================================================================
# ROUTES REST : Gestion de la Configuration et des Caméras
# ==============================================================================

@app.get("/api/config")
async def get_configuration():
    """Retourne la configuration active."""
    return {
        "config": cfg.config,
        "device": detector.device,
        "active_model": detector._active_model_name,
        "open_vocabulary_enabled": detector.open_vocab_enabled,
        "custom_classes": detector.custom_classes,
        "gdpr_notice": GDPR_NOTICE,
    }


@app.post("/api/config", dependencies=[Security(require_enterprise_auth)])
async def update_configuration(payload: Dict[str, Any]):
    """Met à jour les paramètres de détection à la volée (protégé)."""
    if "model_name" in payload:
        new_model = payload["model_name"]
        cfg.set("yolo", "model_name", new_model)
        detector.set_model(new_model)

    if "confidence_threshold" in payload:
        conf = float(payload["confidence_threshold"])
        cfg.set("yolo", "confidence_threshold", conf)
        detector.conf_threshold = conf

    if "open_vocabulary_enabled" in payload:
        open_enabled = bool(payload["open_vocabulary_enabled"])
        custom_classes = payload.get("custom_classes", detector.custom_classes)
        cfg.set("yolo", "open_vocabulary_enabled", open_enabled)
        cfg.set("yolo", "custom_classes", custom_classes)
        detector.set_open_vocabulary(open_enabled, custom_classes)

    if "faces_enabled" in payload:
        cfg.set("faces", "enabled", bool(payload["faces_enabled"]))

    if "pc_camera_index" in payload:
        idx = int(payload["pc_camera_index"])
        cfg.set("camera", "pc_camera_index", idx)
        pc_camera.set_camera_index(idx)

    return {"status": "success", "message": "Configuration mise à jour avec succès"}


# ==============================================================================
# ROUTES REST : Reconnaissance Faciale (RGPD / Données Locales)
# ==============================================================================

@app.get("/api/faces")
async def get_registered_faces():
    """Liste les identités actuellement mémorisées."""
    return {
        "persons": face_engine.list_persons(),
        "count": len(face_engine.known_encodings),
        "gdpr_notice": GDPR_NOTICE,
    }


@app.get("/api/faces/{name}/thumbnail")
async def get_face_thumbnail(name: str):
    """Renvoie la photo miniature d'une personne enregistrée (protégé contre Path Traversal)."""
    try:
        person_dir = safe_resolve_path(face_engine.storage_dir, name)
        thumb_path = person_dir / "thumbnail.jpg"
        if not thumb_path.exists():
            raise HTTPException(status_code=404, detail="Vignette introuvable")
        return FileResponse(thumb_path, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Requête invalide.")


@app.post("/api/faces/register", dependencies=[Security(require_enterprise_auth)])
async def register_face(
    name: str = Form(...),
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
):
    """
    Enregistre un nouveau visage soit par fichier téléversé, soit par capture base64 (protégé).
    """
    img_bgr = None
    if file:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    elif image_base64:
        if "," in image_base64:
            _, b64_part = image_base64.split(",", 1)
        else:
            b64_part = image_base64
        img_bytes = base64.b64decode(b64_part)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Aucune image valide fournie.")

    success, msg = face_engine.register_person(name, img_bgr)
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {"status": "success", "message": msg, "person": name}


@app.delete("/api/faces/{name}", dependencies=[Security(require_enterprise_auth)])
async def delete_face(name: str):
    """Supprime une personne et ses données biométriques (protégé)."""
    if face_engine.delete_person(name):
        return {"status": "success", "message": f"Identité '{name}' supprimée."}
    raise HTTPException(status_code=404, detail="Personne introuvable.")


@app.delete("/api/faces", dependencies=[Security(require_enterprise_auth)])
async def delete_all_faces():
    """Purge intégrale de toutes les données biométriques (protégé)."""
    face_engine.delete_all()
    return {"status": "success", "message": "Toutes les données faciales ont été supprimées avec succès."}



# ==============================================================================
# ROUTES REST : Intelligence & Reconnaissance d'Identité Web
# ==============================================================================

@app.post("/api/web/learn", dependencies=[Security(require_enterprise_auth)])
async def learn_from_web(payload: Dict[str, Any]):
    """
    Apprend automatiquement une personnalité depuis le Web (Wikipédia),
    télécharge sa photo officielle, enregistre son empreinte faciale et sa biographie (protégé).
    """
    name = payload.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Veuillez fournir un nom.")

    loop = asyncio.get_event_loop()
    success, msg, info = await loop.run_in_executor(
        None, web_identity.learn_person_from_web, name, face_engine
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {"status": "success", "message": msg, "info": info}


@app.get("/api/web/info/{name}")
async def get_web_person_info(name: str):
    """Récupère les informations biographiques Wikipédia d'une personne."""
    loop = asyncio.get_event_loop()
    info = await loop.run_in_executor(None, web_identity.get_person_info, name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Aucune fiche trouvée pour '{name}'.")
    return info


@app.post("/api/web/reverse-search")
async def reverse_face_search(payload: Dict[str, Any]):
    """
    Déclenche une recherche visuelle inversée sur Google Lens
    à partir d'une capture de visage inconnu envoyée en base64.
    """
    img_data_url = payload.get("image", "")
    if not img_data_url:
        raise HTTPException(status_code=400, detail="Aucune image de visage fournie.")

    if "," in img_data_url:
        _, b64_part = img_data_url.split(",", 1)
    else:
        b64_part = img_data_url

    try:
        img_bytes = base64.b64decode(b64_part)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Format image invalide.")

        loop = asyncio.get_event_loop()
        saved_path = await loop.run_in_executor(
            None, web_identity.search_face_online, img_bgr, True
        )
        return {
            "status": "success",
            "message": "Recherche Google Lens ouverte dans le navigateur.",
            "file_path": saved_path,
            "lens_url": "https://lens.google.com/",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/faces/{name}/bio", dependencies=[Security(require_enterprise_auth)])
async def update_person_bio(name: str, payload: Dict[str, Any]):
    """Met à jour les métadonnées / notes d'une fiche d'identité (protégé)."""
    if face_engine.save_bio(name, payload):
        return {"status": "success", "message": f"Biographie de '{name}' mise à jour."}
    raise HTTPException(status_code=400, detail="Impossible de sauvegarder la biographie.")


# ==============================================================================
# ROUTES REST : Sûreté Entreprise & Contrôle Comportemental
# ==============================================================================

@app.get("/api/enterprise/config")
async def get_enterprise_config():
    """Renvoie la configuration actuelle du moteur de sûreté d'entreprise."""
    return {
        "restricted_zone_enabled": behavior_engine.restricted_zone_enabled,
        "restricted_zone": behavior_engine.restricted_zone,
        "loitering_threshold_s": behavior_engine.loitering_threshold_s,
        "fall_detection_enabled": behavior_engine.fall_detection_enabled,
        "fast_movement_enabled": behavior_engine.fast_movement_enabled,
        "current_source": pc_camera.camera_index,
        "current_backend": pc_camera.backend_name,
    }


@app.post("/api/enterprise/config", dependencies=[Security(require_enterprise_auth)])
async def update_enterprise_config(payload: Dict[str, Any]):
    """Met à jour les règles et seuils de surveillance comportementale (protégé)."""
    behavior_engine.update_config(payload)
    audit_logger.log_event(
        event_type="CONFIG_CHANGE",
        severity="INFO",
        person_name="Admin",
        description="Mise à jour des règles de sûreté d'entreprise",
    )
    return {"status": "success", "message": "Paramètres de sûreté mis à jour avec succès"}


@app.get("/api/enterprise/audit-logs", dependencies=[Security(require_enterprise_auth)])
async def get_audit_logs(limit: int = 50, severity: str = "ALL"):
    """Récupère l'historique inaltérable des incidents et événements de sécurité (protégé)."""
    logs = audit_logger.get_recent_events(limit=limit, severity_filter=severity)
    return {"status": "success", "count": len(logs), "logs": logs}


@app.get("/api/enterprise/audit-export", dependencies=[Security(require_enterprise_auth)])
async def export_audit_csv():
    """Génère et télécharge le registre d'audit officiel au format CSV (protégé)."""
    csv_data = audit_logger.export_csv()
    return HTMLResponse(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=rapport_audit_surete.csv"},
    )


@app.post("/api/enterprise/rtsp", dependencies=[Security(require_enterprise_auth)])
async def connect_rtsp_camera(payload: Dict[str, Any]):
    """Connecte le système à une caméra IP industrielle (RTSP / ONVIF) (protégé)."""
    url = payload.get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Veuillez fournir une URL de flux valide.")

    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, pc_camera.set_source, url)
    if success:
        audit_logger.log_event(
            event_type="CAMERA_SWITCH",
            severity="INFO",
            person_name="Admin",
            description=f"Connexion au flux caméra IP : {url}",
        )
        return {"status": "success", "message": f"Connecté à la caméra IP : {url}"}
    raise HTTPException(status_code=400, detail=f"Échec de connexion au flux IP : {url}")


@app.get("/api/enterprise/actions/status")
async def get_actions_status():
    """Renvoie l'état des actionneurs (relais de porte, webhook d'urgence, anti-spoofing)."""
    return {
        "status": action_dispatcher.get_status(),
        "liveness_enabled": liveness_detector.enabled,
        "liveness_threshold": liveness_detector.liveness_threshold,
        "api_key_configured": bool(security_manager.api_key),
    }


@app.post("/api/enterprise/actions/door", dependencies=[Security(require_enterprise_auth)])
async def control_door_relay(payload: Dict[str, Any]):
    """Déclenche manuellement le relais de porte ou le confinement (LOCKDOWN) (protégé)."""
    action = payload.get("action", "UNLOCK").upper()
    duration = float(payload.get("duration_s", 5.0))
    res = action_dispatcher.trigger_door_relay(action, duration_s=duration)
    audit_logger.log_event(
        event_type="DOOR_CONTROL",
        severity="INFO" if action == "UNLOCK" else "WARNING",
        person_name="Admin",
        description=f"Commande manuelle relais : {action}",
    )
    return {"status": "success", "action": res}


@app.post("/api/enterprise/actions/webhook", dependencies=[Security(require_enterprise_auth)])
async def configure_emergency_webhook(payload: Dict[str, Any]):
    """Configure l'URL de notification d'incident pour SIEM / Teams / Slack (protégé)."""
    url = payload.get("url", "")
    action_dispatcher.set_webhook_url(url)
    return {"status": "success", "message": "Webhook d'incident mis à jour avec succès"}


@app.post("/api/enterprise/liveness", dependencies=[Security(require_enterprise_auth)])
async def configure_liveness(payload: Dict[str, Any]):
    """Active ou désactive la détection de vivant (Anti-Spoofing) (protégé)."""
    if "enabled" in payload:
        liveness_detector.enabled = bool(payload["enabled"])
    if "threshold" in payload:
        liveness_detector.liveness_threshold = float(payload["threshold"])
    return {
        "status": "success",
        "enabled": liveness_detector.enabled,
        "threshold": liveness_detector.liveness_threshold,
    }


# ==============================================================================
# ROUTES REST : Captures d'écran
# ==============================================================================

@app.post("/api/snapshots/save")
async def save_snapshot(payload: Dict[str, Any]):
    """Sauvegarde une capture d'écran annotée sur le disque."""
    img_data_url = payload.get("image", "")
    if "," in img_data_url:
        _, b64_part = img_data_url.split(",", 1)
    else:
        b64_part = img_data_url

    img_bytes = base64.b64decode(b64_part)
    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    target_path = CAPTURES_DIR / filename
    with open(target_path, "wb") as f:
        f.write(img_bytes)

    return {
        "status": "success",
        "filename": filename,
        "url": f"/api/captures/{filename}",
    }


@app.get("/api/captures")
async def list_captures():
    """Liste les captures d'écran enregistrées."""
    files = sorted(CAPTURES_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "captures": [
            {
                "filename": f.name,
                "url": f"/api/captures/{f.name}",
                "timestamp": datetime.fromtimestamp(f.stat().st_mtime).strftime("%d/%m/%Y %H:%M:%S"),
            }
            for f in files
        ]
    }


@app.get("/api/captures/{filename}")
async def get_capture(filename: str):
    """Renvoie une capture spécifique (protégé contre Path Traversal)."""
    try:
        file_path = safe_resolve_path(CAPTURES_DIR, filename)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Fichier introuvable.")
        return FileResponse(file_path, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Nom de fichier invalide.")


@app.get("/api/captures/osint/{filename}")
async def get_osint_capture(filename: str):
    """Renvoie une capture de cible OSINT (protégé contre Path Traversal)."""
    try:
        file_path = safe_resolve_path(osint_investigator.osint_dir, filename)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Fichier cible introuvable.")
        return FileResponse(file_path, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Nom de fichier cible invalide.")



# ==============================================================================
# ROUTES REST : Investigation OSINT & Profiling Réseaux Sociaux
# ==============================================================================

@app.post("/api/osint/target")
async def prepare_osint_target(payload: Dict[str, Any]):
    """
    Extrait le visage inconnu depuis une capture webcam,
    le prépare et renvoie les liens vers les 5 moteurs de recherche inversée.
    """
    img_data_url = payload.get("image", "")
    if not img_data_url:
        raise HTTPException(status_code=400, detail="Aucune image de visage fournie.")

    if "," in img_data_url:
        _, b64_part = img_data_url.split(",", 1)
    else:
        b64_part = img_data_url

    try:
        img_bytes = base64.b64decode(b64_part)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Décodage image impossible.")

        loop = asyncio.get_event_loop()
        dossier = await loop.run_in_executor(
            None, osint_investigator.prepare_target_investigation, img_bgr
        )
        return dossier
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/osint/dossier/{name}")
async def get_osint_social_dossier(name: str):
    """Génère le dossier de recherche réseaux sociaux pour une personne."""
    loop = asyncio.get_event_loop()
    dossier = await loop.run_in_executor(
        None, osint_investigator.generate_social_dossier, name
    )
    return dossier


@app.post("/api/osint/enroll", dependencies=[Security(require_enterprise_auth)])
async def enroll_osint_profile(payload: Dict[str, Any]):
    """
    Associe un nom et des profils trouvés à l'image capturée
    pour que la webcam le reconnaisse désormais en direct.
    """
    name = payload.get("name", "").strip()
    role = payload.get("role", "").strip()
    notes = payload.get("notes", "").strip()
    image_path = payload.get("image_path", "").strip()

    if not name or not image_path:
        raise HTTPException(status_code=400, detail="Le nom et le chemin de l'image sont obligatoires.")

    loop = asyncio.get_event_loop()
    success, msg = await loop.run_in_executor(
        None,
        osint_investigator.register_investigated_profile,
        name,
        role,
        notes,
        image_path,
        face_engine,
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {"status": "success", "message": msg}


# ==============================================================================
# Interface Web & Fichiers Statiques
# ==============================================================================

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Point d'entrée web principal."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Vision IA : Initialisation des fichiers statiques...</h1>")
