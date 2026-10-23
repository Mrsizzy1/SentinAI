"""
Application Desktop native (fenêtre OpenCV) pour la Vision IA Temps Réel.
Affiche directement le flux webcam du PC avec détections YOLOv8,
reconnaissance faciale et raccourcis clavier interactifs.

Usage :
    python desktop.py
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

from datetime import datetime
import time
import cv2
import numpy as np

# Reconfiguration UTF-8 console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.audit import audit_logger
from app.behavior import behavior_engine
from app.camera_stream import PCCameraStream
from app.config import cfg
from app.detector import ObjectDetector
from app.faces import FaceEngine
from app.osint_investigator import osint_investigator
from app.web_identity import web_identity


def draw_bottom_bar(frame: np.ndarray, backend_name: str, cam_idx: int, notification: str = "") -> np.ndarray:
    """Dessine une barre d'état et de raccourcis en bas de l'image."""
    h, w = frame.shape[:2]
    bar_height = 36
    bar_y = h - bar_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (18, 20, 26), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    if notification:
        # Message de notification temporaire (ex: Capture sauvegardée)
        cv2.putText(
            frame,
            notification,
            (15, h - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (50, 240, 100),
            1,
            cv2.LINE_AA,
        )
    else:
        zone_status = "ON" if behavior_engine.restricted_zone_enabled else "OFF"
        shortcuts = f"[Q] Quitter  [S] Capture  [W] Enquete OSINT  [Z] Zone:{zone_status}  [B] Backend:{backend_name}  [C] Cam#{cam_idx}"
        cv2.putText(
            frame,
            shortcuts,
            (15, h - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (210, 215, 225),
            1,
            cv2.LINE_AA,
        )

    return frame


def render_black_screen_help(frame: np.ndarray, cam_idx: int, backend_name: str) -> np.ndarray:
    """Affiche des consignes claires si la caméra produit une image noire."""
    h, w = frame.shape[:2]
    box_w, box_h = min(580, w - 40), 160
    bx1 = (w - box_w) // 2
    by1 = (h - box_h) // 2
    bx2 = bx1 + box_w
    by2 = by1 + box_h

    # Boîte semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (25, 25, 35), -1)
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (40, 140, 255), 2)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

    # Titre d'alerte
    cv2.putText(
        frame,
        "ATTENTION : Image noire detectee",
        (bx1 + 18, by1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 200, 255),
        2,
        cv2.LINE_AA,
    )

    lines = [
        "1. Ouvrez le clapet / volet mecanique sur la lentille de votre webcam",
        "2. Verifiez la touche camera du clavier HP (F9 ou Fn+F9)",
        f"3. Appuyez sur [B] pour changer de backend (Actuel: {backend_name})",
        f"4. Appuyez sur [C] pour tester l'autre camera (Actuelle: #{cam_idx})",
    ]

    y_offset = by1 + 60
    for line in lines:
        cv2.putText(
            frame,
            line,
            (bx1 + 18, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 240),
            1,
            cv2.LINE_AA,
        )
        y_offset += 24

    return frame


def run_desktop():
    """Point d'entrée de l'application Desktop."""
    print("\n" + "=" * 65)
    print("   [VISION IA TEMPS RÉEL] - APPLICATION DESKTOP (OPENCV)")
    print("=" * 65)

    # 1. Initialisation des composants
    cam_index = int(cfg.get("camera", "pc_camera_index", default=0))
    width = int(cfg.get("camera", "width", default=640))
    height = int(cfg.get("camera", "height", default=480))

    print(f"[*] Démarrage de la webcam PC index #{cam_index}...")
    camera = PCCameraStream(camera_index=cam_index, width=width, height=height)
    if not camera.start():
        print(f"[ERREUR] Impossible d'ouvrir la webcam PC index #{cam_index}.")
        print("          Vérifiez qu'aucune autre application n'utilise votre caméra.")
        return

    print("[*] Initialisation du détecteur YOLO...")
    detector = ObjectDetector(
        model_name=cfg.get("yolo", "model_name", default="yolov8n.pt"),
        conf_threshold=float(cfg.get("yolo", "confidence_threshold", default=0.45)),
        iou_threshold=float(cfg.get("yolo", "iou_threshold", default=0.45)),
    )

    print("[*] Initialisation de la reconnaissance faciale...")
    faces_enabled = cfg.get("faces", "enabled", default=True)
    face_engine = FaceEngine(
        storage_dir=str(BASE_DIR / "data" / "faces"),
        tolerance=float(cfg.get("faces", "tolerance", default=0.52)),
    )

    captures_dir = BASE_DIR / "data" / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    # 2. Fenêtre OpenCV
    win_title = "Vision IA Temps Reel - Desktop"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 960, 720)

    print("[OK] Fenêtre Desktop ouverte avec succès !")
    print("     Raccourcis : [Q]=Quitter | [W]=Enquête OSINT Visage/Web | [Z]=Zone Sûreté | [B]=Backend | [C]=Caméra | [S]=Capture | [P]=Pause\n")

    is_paused = False
    is_fullscreen = False
    notification_text = ""
    notification_until = 0.0

    last_time = time.perf_counter()
    fps = 0.0
    models_cycle = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

    try:
        while True:
            if not is_paused:
                ret, frame = camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Calcul du FPS lissé
                now = time.perf_counter()
                dt = now - last_time
                if dt > 0:
                    fps = 0.85 * fps + 0.15 * (1.0 / dt) if fps > 0 else (1.0 / dt)
                last_time = now

                # Vérification de luminosité de l'image
                brightness = float(np.mean(frame))

                # Inférence YOLO
                detections, _ = detector.detect(frame)

                # Reconnaissance faciale
                if faces_enabled:
                    detections = face_engine.attach_face_ids_to_detections(frame, detections)

                # Mémorisation de la découpe du dernier visage inconnu pour la recherche Web
                last_unknown_face_crop = None
                for det in detections:
                    if det.get("category") == "human" and det.get("face_name") == "Inconnu":
                        x1, y1, x2, y2 = det["box"]
                        face_y2 = y1 + int((y2 - y1) * 0.55)
                        x1_c, y1_c = max(0, x1), max(0, y1)
                        x2_c, face_y2_c = min(frame.shape[1], x2), min(frame.shape[0], face_y2)
                        if (x2_c - x1_c) >= 30 and (face_y2_c - y1_c) >= 30:
                            last_unknown_face_crop = frame[y1_c:face_y2_c, x1_c:x2_c].copy()

                # Analyse Comportementale Entreprise & Sûreté
                active_alerts, b_summary = behavior_engine.analyze(frame, detections, audit_logger)

                # Annotation visuelle (boîtes, zone restreinte et alertes)
                annotated = detector.annotate_frame(
                    frame,
                    detections,
                    fps=fps,
                    behavior_summary=b_summary,
                    active_alerts=active_alerts,
                )

                # Si l'image est noire, afficher l'aide interactive
                if brightness < 1.5:
                    annotated = render_black_screen_help(annotated, camera.camera_index, camera.backend_name)

                # Bandeau d'actions inférieur
                notif = notification_text if time.time() < notification_until else ""
                display_frame = draw_bottom_bar(annotated, camera.backend_name, camera.camera_index, notif)
            else:
                # Affichage indicateur PAUSE
                overlay = display_frame.copy()
                h, w = overlay.shape[:2]
                cv2.putText(
                    overlay,
                    "PAUSE (Appuyez sur 'P' pour reprendre)",
                    (w // 2 - 230, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                display_frame = overlay

            cv2.imshow(win_title, display_frame)

            # Gestion des touches clavier (délai 1ms)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # Touche Q ou Échap
                break

            elif key in (ord("s"), ord("S")):  # Touche S (Capture)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_desktop_{ts}.jpg"
                save_path = captures_dir / filename
                cv2.imwrite(str(save_path), annotated)
                notification_text = f"Capture sauvegardee : {filename}"
                notification_until = time.time() + 2.5
                print(f"[Capture] Sauvegardée -> {save_path}")

            elif key in (ord("w"), ord("W")):  # Touche W (Enquête OSINT Visage / Recherche Réseaux Sociaux)
                target_img = last_unknown_face_crop if last_unknown_face_crop is not None else frame
                saved_path = web_identity.search_face_online(target_img, open_browser=True)
                osint_data = osint_investigator.prepare_target_investigation(target_img)
                target_desc = "visage inconnu" if last_unknown_face_crop is not None else "image actuelle"
                notification_text = f"Enquete OSINT lancee pour {target_desc} !"
                notification_until = time.time() + 4.0
                print("\n" + "=" * 68)
                print("   [*] ENQUETE OSINT & RECHERCHE IDENTITE WEB LANCEE")
                print("=" * 68)
                print(f"[Cible isolée] : {osint_data['file_path']}")
                print("[1] Google Lens      : ouvert dans le navigateur web.")
                print(f"[2] Yandex Visual    : {osint_data['search_engines']['yandex_images']['url']}")
                print(f"[3] FaceCheck.ID     : {osint_data['search_engines']['facecheck_id']['url']}")
                print("[-] Dashboard Web    : http://localhost:8000 -> Cliquez sur [Enquete OSINT]")
                print("    pour associer le nom trouve, ses reseaux sociaux, et l'enroler en 1 clic !")
                print("=" * 68 + "\n")

            elif key in (ord("z"), ord("Z")):  # Touche Z (Bascule Zone Restreinte Entreprise)
                behavior_engine.restricted_zone_enabled = not behavior_engine.restricted_zone_enabled
                st = "ACTIVÉE" if behavior_engine.restricted_zone_enabled else "DÉSACTIVÉE"
                notification_text = f"Zone de sûreté restreinte : {st}"
                notification_until = time.time() + 2.5
                print(f"[Zone] Sûreté restreinte -> {st}")

            elif key in (ord("b"), ord("B")):  # Touche B (Changer backend MSMF <-> DSHOW)
                camera.cycle_backend()
                notification_text = f"Nouveau backend : {camera.backend_name}"
                notification_until = time.time() + 2.5
                print(f"[Backend] Bascule sur : {camera.backend_name}")

            elif key in (ord("c"), ord("C")):  # Touche C (Changer d'index de caméra)
                next_cam = 1 if camera.camera_index == 0 else 0
                camera.set_camera_index(next_cam)
                notification_text = f"Bascule vers Camera #{next_cam}"
                notification_until = time.time() + 2.5
                print(f"[Camera] Bascule vers index #{next_cam}")

            elif key in (ord("p"), ord("P")):  # Touche P (Pause)
                is_paused = not is_paused

            elif key in (ord("f"), ord("F")):  # Touche F (Plein écran)
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(win_title, cv2.WND_PROP_FULLSCREEN, prop)

            elif key in (ord("m"), ord("M")):  # Touche M (Changement de modèle)
                current_idx = models_cycle.index(detector.model_name) if detector.model_name in models_cycle else 0
                next_model = models_cycle[(current_idx + 1) % len(models_cycle)]
                detector.set_model(next_model)
                notification_text = f"Modele actif : {next_model}"
                notification_until = time.time() + 2.0
                print(f"[Modele] Bascule vers {next_model}")

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[*] Fermeture de l'application Desktop...")
        camera.stop()
        cv2.destroyAllWindows()
        print("[OK] Fermeture effectuee proprement.")


if __name__ == "__main__":
    run_desktop()
