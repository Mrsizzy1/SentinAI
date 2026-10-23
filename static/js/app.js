/**
 * Vision IA Temps Réel - Client Frontend (PC & Mobile)
 * Gestion du streaming vidéo, WebSockets, getUserMedia, YOLO-World et reconnaissance faciale.
 */

// Intercepteur Fetch pour inclure automatiquement la clé d'API Entreprise si configurée
const originalFetch = window.fetch;
window.fetch = function (url, options = {}) {
    const apiKey = localStorage.getItem("enterprise_api_key") || "";
    if (apiKey) {
        options = options || {};
        options.headers = options.headers || {};
        if (options.headers instanceof Headers) {
            if (!options.headers.has("X-Enterprise-Key")) {
                options.headers.set("X-Enterprise-Key", apiKey);
            }
        } else if (Array.isArray(options.headers)) {
            options.headers.push(["X-Enterprise-Key", apiKey]);
        } else {
            if (!options.headers["X-Enterprise-Key"]) {
                options.headers["X-Enterprise-Key"] = apiKey;
            }
        }
    }
    return originalFetch(url, options);
};

// ==============================================================================
// État Global de l'Application
// ==============================================================================

const isMobileDevice = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

const state = {
    mode: "browser",            // "browser" (caméra smartphone/client) ou "pc" (webcam OpenCV serveur)
    facingMode: isMobileDevice ? "environment" : "user",  // "environment" (arrière mobile) ou "user" (webcam PC/selfie)
    isMobile: isMobileDevice,
    streamActive: false,
    isPaused: false,
    awaitingServerResponse: false,
    socket: null,
    videoTrack: null,
    lastFrameSentTimestamp: 0,
    currentAnnotatedImage: "",
    lastDetections: [],         // Dernières détections reçues du serveur
    osintLastImagePath: null,   // Chemin de l'image OSINT capturée
    osintFoundName: "",         // Nom identifié par l'enquêteur
};

// Éléments DOM
const elements = {
    statusDot: document.getElementById("connectionStatusDot"),
    hardwareBadge: document.getElementById("hardwareBadge"),
    activeModelBadge: document.getElementById("activeModelBadge"),
    deviceSubtitle: document.getElementById("deviceSubtitle"),

    webcamVideo: document.getElementById("webcamVideo"),
    captureCanvas: document.getElementById("captureCanvas"),
    annotatedStream: document.getElementById("annotatedStream"),

    hudFps: document.getElementById("hudFps"),
    hudLatency: document.getElementById("hudLatency"),
    hudObjects: document.getElementById("hudObjects"),
    categoriesChips: document.getElementById("categoriesChips"),

    btnSwitchCamera: document.getElementById("btnSwitchCamera"),
    btnSnapshot: document.getElementById("btnSnapshot"),
    btnTogglePause: document.getElementById("btnTogglePause"),

    radioSources: document.querySelectorAll("input[name='cameraSource']"),
    pcCameraIndexGroup: document.getElementById("pcCameraIndexGroup"),
    pcCamIndexInput: document.getElementById("pcCamIndex"),
    btnApplyCamIndex: document.getElementById("btnApplyCamIndex"),

    selectModel: document.getElementById("selectModel"),
    rangeConfidence: document.getElementById("rangeConfidence"),
    confVal: document.getElementById("confVal"),
    checkFacesEnabled: document.getElementById("checkFacesEnabled"),

    toggleOpenVocab: document.getElementById("toggleOpenVocab"),
    customClassesInput: document.getElementById("customClassesInput"),
    btnApplyOpenVocab: document.getElementById("btnApplyOpenVocab"),

    facePersonName: document.getElementById("facePersonName"),
    btnCaptureCurrentFace: document.getElementById("btnCaptureCurrentFace"),
    faceFileInput: document.getElementById("faceFileInput"),
    facesGallery: document.getElementById("facesGallery"),
    facesCountBadge: document.getElementById("facesCountBadge"),
    btnPurgeFaces: document.getElementById("btnPurgeFaces"),

    webPersonName: document.getElementById("webPersonName"),
    btnLearnWebPerson: document.getElementById("btnLearnWebPerson"),
    webBioPreview: document.getElementById("webBioPreview"),
    webBioThumb: document.getElementById("webBioThumb"),
    webBioTitle: document.getElementById("webBioTitle"),
    webBioDesc: document.getElementById("webBioDesc"),
    webBioExtract: document.getElementById("webBioExtract"),
    webBioLink: document.getElementById("webBioLink"),

    btnWebSearchFace: document.getElementById("btnWebSearchFace"),
    capturesGallery: document.getElementById("capturesGallery"),
    btnRefreshCaptures: document.getElementById("btnRefreshCaptures"),
    toast: document.getElementById("toastNotification"),

    checkRestrictedZone: document.getElementById("checkRestrictedZone"),
    checkFallDetection: document.getElementById("checkFallDetection"),
    inputLoitering: document.getElementById("inputLoitering"),
    btnSaveBehaviorConfig: document.getElementById("btnSaveBehaviorConfig"),
    inputRtspUrl: document.getElementById("inputRtspUrl"),
    btnConnectRtsp: document.getElementById("btnConnectRtsp"),
    btnRefreshAudit: document.getElementById("btnRefreshAudit"),
    auditTableBody: document.getElementById("auditTableBody"),

    checkLiveness: document.getElementById("checkLiveness"),
    btnDoorUnlock: document.getElementById("btnDoorUnlock"),
    btnDoorLockdown: document.getElementById("btnDoorLockdown"),
    doorRelayStatusBadge: document.getElementById("doorRelayStatusBadge"),
    inputWebhookUrl: document.getElementById("inputWebhookUrl"),
    btnSaveWebhook: document.getElementById("btnSaveWebhook"),
};

// ==============================================================================
// Utilitaires & Notifications Toast
// ==============================================================================

function showToast(message, type = "success") {
    elements.toast.textContent = message;
    elements.toast.className = `toast show toast-${type}`;
    setTimeout(() => {
        elements.toast.className = "toast";
    }, 3200);
}

function getWebSocketUrl(endpoint) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const apiKey = localStorage.getItem("enterprise_api_key") || "";
    const tokenParam = apiKey ? `?token=${encodeURIComponent(apiKey)}` : "";
    return `${protocol}//${window.location.host}${endpoint}${tokenParam}`;
}

// ==============================================================================
// Initialisation & Configuration Serveur
// ==============================================================================

async function loadServerConfig() {
    try {
        const res = await fetch("/api/config");
        if (!res.ok) return;
        const data = await res.json();

        // Affichage matériel (CUDA / MPS / CPU)
        const deviceMap = { cuda: "GPU (CUDA)", mps: "GPU (Apple Silicon)", cpu: "CPU" };
        elements.hardwareBadge.textContent = deviceMap[data.device] || data.device.toUpperCase();

        // Modèle actif
        elements.activeModelBadge.textContent = data.open_vocabulary_enabled
            ? "YOLO-World (Ouvert)"
            : data.config.yolo.model_name;

        // Remplissage des formulaires
        elements.selectModel.value = data.config.yolo.model_name;
        elements.rangeConfidence.value = data.config.yolo.confidence_threshold;
        elements.confVal.textContent = `${Math.round(data.config.yolo.confidence_threshold * 100)}%`;
        elements.checkFacesEnabled.checked = data.config.faces.enabled;

        elements.toggleOpenVocab.checked = data.open_vocabulary_enabled;
        elements.customClassesInput.value = (data.custom_classes || []).join(", ");

        elements.pcCamIndexInput.value = data.config.camera.pc_camera_index;
    } catch (e) {
        console.error("Erreur chargement config:", e);
    }
}

// ==============================================================================
// Gestion de la Caméra Téléphone / Navigateur (getUserMedia)
// ==============================================================================

async function startBrowserCamera() {
    stopBrowserCamera();

    let stream = null;

    // 1ère tentative : avec résolution et orientation préférée
    const preferredConstraints = {
        audio: false,
        video: {
            facingMode: state.isMobile ? { ideal: state.facingMode } : "user",
            width: { ideal: 640 },
            height: { ideal: 480 },
        },
    };

    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("getUserMedia non supporté ou contexte non sécurisé");
        }
        stream = await navigator.mediaDevices.getUserMedia(preferredConstraints);
    } catch (err1) {
        console.warn("Échec contraintes idéales, tentative générique...", err1);
        // 2ème tentative de secours : contraintes minimales (accepte n'importe quelle webcam disponible)
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
        } catch (err2) {
            console.error("Accès caméra impossible :", err2);
            showToast("Impossible d'accéder à la caméra du navigateur. Autorisez la caméra ou essayez la source 'Webcam PC (Serveur)'.", "error");
            return;
        }
    }

    try {
        elements.webcamVideo.srcObject = stream;
        state.videoTrack = stream.getVideoTracks()[0];
        await elements.webcamVideo.play();

        // Ajustement des dimensions de capture
        elements.captureCanvas.width = elements.webcamVideo.videoWidth || 640;
        elements.captureCanvas.height = elements.webcamVideo.videoHeight || 480;

        state.streamActive = true;
        initWebSocketStream();
    } catch (err) {
        console.error("Erreur démarrage vidéo :", err);
        showToast("Erreur lors de la lecture du flux vidéo.", "error");
    }
}

function stopBrowserCamera() {
    state.streamActive = false;
    if (state.videoTrack) {
        state.videoTrack.stop();
        state.videoTrack = null;
    }
    if (elements.webcamVideo.srcObject) {
        elements.webcamVideo.srcObject.getTracks().forEach((t) => t.stop());
        elements.webcamVideo.srcObject = null;
    }
}

// ==============================================================================
// Gestion des WebSockets & Boucle Vidéo Temps Réel
// ==============================================================================

function initWebSocketStream() {
    if (state.socket) {
        state.socket.close();
        state.socket = null;
    }

    const endpoint = state.mode === "browser" ? "/ws/phone_stream" : "/ws/pc_stream";
    const wsUrl = getWebSocketUrl(endpoint);
    state.socket = new WebSocket(wsUrl);

    state.socket.onopen = () => {
        elements.statusDot.style.backgroundColor = "var(--accent-emerald)";
        elements.statusDot.classList.add("pulsing");
        state.awaitingServerResponse = false;

        if (state.mode === "browser") {
            sendNextFrame();
        }
    };

    state.socket.onmessage = (event) => {
        state.awaitingServerResponse = false;
        try {
            const data = JSON.parse(event.data);
            if (data.image) {
                elements.annotatedStream.src = data.image;
                state.currentAnnotatedImage = data.image;
            }

            // Mise à jour du HUD
            elements.hudFps.textContent = (data.fps || 0).toFixed(1);
            elements.hudLatency.textContent = `${Math.round(data.latency_ms || 0)} ms`;
            elements.hudObjects.textContent = data.detections_count || 0;

            // Mise à jour des catégories détectées
            updateCategoriesChips(data.detections || [], data.categories || {});
        } catch (e) {
            console.error("Erreur payload WebSocket:", e);
        }

        // Si mode mobile/navigateur : envoie l'image suivante dès que le serveur a répondu
        if (state.mode === "browser" && state.streamActive && !state.isPaused) {
            requestAnimationFrame(sendNextFrame);
        }
    };

    state.socket.onclose = () => {
        elements.statusDot.style.backgroundColor = "var(--accent-red)";
        elements.statusDot.classList.remove("pulsing");
    };

    state.socket.onerror = (err) => {
        console.error("Erreur WebSocket:", err);
    };
}

function sendNextFrame() {
    if (!state.streamActive || state.isPaused || state.awaitingServerResponse) return;
    if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;

    const video = elements.webcamVideo;
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
        requestAnimationFrame(sendNextFrame);
        return;
    }

    const ctx = elements.captureCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, 640, 480);

    // Encodage JPEG léger pour optimiser la bande passante Wi-Fi
    const jpegData = elements.captureCanvas.toDataURL("image/jpeg", 0.72);

    state.awaitingServerResponse = true;
    state.socket.send(JSON.stringify({ image: jpegData }));
}

// ==============================================================================
// Mise à Jour Dynamique des Catégories & Badges HUD
// ==============================================================================

function updateCategoriesChips(detections, categories) {
    elements.categoriesChips.innerHTML = "";

    if (detections.length === 0) {
        elements.categoriesChips.innerHTML = '<span class="chip chip-empty">Aucun objet détecté</span>';
        return;
    }

    // Identités de visages reconnus
    const humans = detections.filter((d) => d.category === "human");
    for (const h of humans) {
        const chip = document.createElement("span");
        chip.className = "chip chip-human";
        const name = h.face_name || "Humain";
        if (h.face_desc) {
            chip.textContent = `👤 ${name} (${h.face_desc})`;
        } else {
            chip.textContent = `👤 ${name}`;
        }
        elements.categoriesChips.appendChild(chip);
    }

    // Animaux
    const animals = detections.filter((d) => d.category === "animal");
    for (const a of animals) {
        const chip = document.createElement("span");
        chip.className = "chip chip-animal";
        chip.textContent = `🐾 ${a.label}`;
        elements.categoriesChips.appendChild(chip);
    }

    // Objets
    const objects = detections.filter((d) => d.category === "object");
    const counts = {};
    for (const o of objects) {
        counts[o.label] = (counts[o.label] || 0) + 1;
    }

    for (const [label, count] of Object.entries(counts)) {
        const chip = document.createElement("span");
        chip.className = "chip chip-object";
        chip.textContent = `📦 ${label} (${count})`;
        elements.categoriesChips.appendChild(chip);
    }
}

// ==============================================================================
// Contrôles Flottants Vidéo (Flip, Pause, Capture)
// ==============================================================================

elements.btnSwitchCamera.addEventListener("click", async () => {
    state.facingMode = state.facingMode === "environment" ? "user" : "environment";
    showToast(`Caméra basculée : ${state.facingMode === "user" ? "Avant (Selfie)" : "Arrière"}`);
    if (state.mode === "browser") {
        await startBrowserCamera();
    }
});

elements.btnTogglePause.addEventListener("click", () => {
    state.isPaused = !state.isPaused;
    elements.btnTogglePause.textContent = state.isPaused ? "▶️" : "⏸️";
    if (!state.isPaused && state.mode === "browser") {
        sendNextFrame();
    }
});

elements.btnSnapshot.addEventListener("click", async () => {
    if (!state.currentAnnotatedImage) {
        showToast("Aucune image disponible pour la capture.", "error");
        return;
    }

    try {
        const res = await fetch("/api/snapshots/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: state.currentAnnotatedImage }),
        });
        const data = await res.json();
        if (data.status === "success") {
            showToast("📸 Capture sauvegardée avec succès !");
            loadCaptures();
        }
    } catch (e) {
        showToast("Erreur lors de la capture.", "error");
    }
});

// ==============================================================================
// Bouton 🌐 - Enquête OSINT & Recherche Identité (Webcam -> Réseaux Sociaux)
// ==============================================================================

/**
 * Ouvre ou crée la modale OSINT affichant :
 * 1. La photo isolée du visage/cible
 * 2. Les boutons de recherche inversée (Google Lens, Yandex, FaceCheck, Bing)
 * 3. La recherche de profils sociaux par nom (LinkedIn, Instagram, Facebook, X, GitHub)
 * 4. L'enrôlement 1-clic pour mémorisation biométrique permanente
 */
function openOsintModal(osintData) {
    // Supprime l'ancienne modale si elle existe
    let modal = document.getElementById("osintModal");
    if (modal) modal.remove();

    const engines = osintData.search_engines || {};
    const imgSrc = osintData.image_url || "";
    state.osintLastImagePath = osintData.file_path || "";

    modal = document.createElement("div");
    modal.id = "osintModal";
    modal.style.cssText = `
        position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 9999;
        background: rgba(8, 10, 18, 0.95); backdrop-filter: blur(8px);
        display: flex; flex-direction: column; align-items: center;
        justify-content: flex-start; overflow-y: auto; padding: 20px;
        font-family: 'Plus Jakarta Sans', sans-serif;
    `;

    modal.innerHTML = `
    <div style="max-width: 700px; width: 100%; background: #12151e; border: 1px solid #2d3555;
        border-radius: 20px; padding: 30px; box-shadow: 0 0 60px rgba(99,102,241,0.3);">

        <!-- En-tête -->
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:24px;">
            <div>
                <h2 style="color:#e2e8f0; margin:0; font-size:1.4rem;">🔍 Enquête OSINT — Identification Web</h2>
                <p style="color:#64748b; margin:4px 0 0; font-size:0.85rem;">Recherche de l'identité en ligne à partir de l'image capturée par la webcam</p>
            </div>
            <button id="btnCloseOsint" style="background:none;border:1px solid #3f4566;border-radius:8px;
                color:#94a3b8;padding:6px 14px;cursor:pointer;font-size:1rem;">✕ Fermer</button>
        </div>

        <!-- ÉTAPE 1: Image capturée + Moteurs de recherche visuelle -->
        <div style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:24px;">
            <div style="flex:0 0 180px; text-align:center;">
                <p style="color:#94a3b8; font-size:0.78rem; margin:0 0 8px;">🎯 CIBLE ISOLÉE</p>
                <img id="osintTargetImg" src="${imgSrc}" alt="Cible" style="width:100%;border-radius:12px;
                    border:2px solid #6366f1; object-fit:cover; max-height:200px;"/>
                <p style="color:#475569; font-size:0.7rem; margin:6px 0 0;">${osintData.timestamp || ''}</p>
            </div>
            <div style="flex:1; min-width:280px;">
                <p style="color:#94a3b8; font-size:0.82rem; margin:0 0 12px;">ÉTAPE 1 — Faites glisser la photo ci-contre dans le moteur de votre choix :</p>
                <div style="display:flex; flex-direction:column; gap:8px;">
                    ${engines.google_lens ? `<a href="${engines.google_lens.url}" target="_blank" style="${btnOsintStyle('#4285f4')}">
                        🔍 Google Lens — Recherche Visuelle Globale</a>` : ''}
                    ${engines.yandex_images ? `<a href="${engines.yandex_images.url}" target="_blank" style="${btnOsintStyle('#e44a2a')}">
                        🌐 Yandex Visual — Meilleur pour les visages</a>` : ''}
                    ${engines.facecheck_id ? `<a href="${engines.facecheck_id.url}" target="_blank" style="${btnOsintStyle('#10b981')}">
                        🕵️ FaceCheck.ID — Spécialisé réseaux sociaux</a>` : ''}
                    ${engines.bing_visual ? `<a href="${engines.bing_visual.url}" target="_blank" style="${btnOsintStyle('#0078d4')}">
                        🔵 Bing Visual Search — Microsoft</a>` : ''}
                </div>
            </div>
        </div>

        <!-- ÉTAPE 2: Profiling réseaux sociaux par nom -->
        <div style="border-top:1px solid #1e2740; padding-top:20px; margin-bottom:20px;">
            <p style="color:#94a3b8; font-size:0.82rem; margin:0 0 10px;">ÉTAPE 2 — Une fois le nom trouvé, découvrez tous ses profils sociaux :</p>
            <div style="display:flex; gap:8px;">
                <input type="text" id="osintModalNameInput" placeholder="Entrez le nom découvert (Ex: Jean Dupont)"
                    style="flex:1; background:#1a1d2e; border:1px solid #2d3555; border-radius:10px;
                    padding:10px 14px; color:#e2e8f0; font-size:0.88rem; outline:none;"/>
                <button id="btnOsintSearchSocial" style="background:#6366f1; border:none; border-radius:10px;
                    color:white; padding:10px 18px; cursor:pointer; font-weight:600;">🔎 Chercher</button>
            </div>
            <div id="osintSocialResults" style="display:none; margin-top:14px; display:grid;
                grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap:8px;"></div>
        </div>

        <!-- ÉTAPE 3: Enrôlement biométrique 1-clic -->
        <div style="border-top:1px solid #1e2740; padding-top:20px;">
            <p style="color:#94a3b8; font-size:0.82rem; margin:0 0 10px;">ÉTAPE 3 — Mémorisez cette personne pour que la caméra la reconnaisse EN DIRECT :</p>
            <div style="display:flex; flex-direction:column; gap:10px;">
                <input type="text" id="osintEnrollName" placeholder="Nom complet (Ex: Jean Dupont)" value=""
                    style="background:#1a1d2e; border:1px solid #2d3555; border-radius:10px;
                    padding:10px 14px; color:#e2e8f0; font-size:0.88rem; outline:none;"/>
                <input type="text" id="osintEnrollRole" placeholder="Rôle / Titre (Ex: Directeur, Suspect, Visiteur...)"
                    style="background:#1a1d2e; border:1px solid #2d3555; border-radius:10px;
                    padding:10px 14px; color:#e2e8f0; font-size:0.88rem; outline:none;"/>
                <button id="btnOsintEnroll" style="background:linear-gradient(135deg,#6366f1,#8b5cf6);
                    border:none; border-radius:10px; color:white; padding:12px; cursor:pointer;
                    font-weight:700; font-size:0.95rem;">
                    💾 Enregistrer dans la caméra — Reconnaissance immédiate
                </button>
                <div id="osintEnrollStatus" style="color:#10b981; font-size:0.82rem; display:none;"></div>
            </div>
        </div>
    </div>`;

    document.body.appendChild(modal);

    // Fermeture
    document.getElementById("btnCloseOsint").addEventListener("click", () => modal.remove());
    modal.addEventListener("click", (e) => { if (e.target === modal) modal.remove(); });

    // Recherche profils sociaux
    const btnSearchSocial = document.getElementById("btnOsintSearchSocial");
    const osintNameInput = document.getElementById("osintModalNameInput");
    const socialResults = document.getElementById("osintSocialResults");
    const enrollNameInput = document.getElementById("osintEnrollName");

    btnSearchSocial.addEventListener("click", async () => {
        const name = osintNameInput.value.trim();
        if (!name) { osintNameInput.focus(); return; }
        enrollNameInput.value = name;
        state.osintFoundName = name;
        btnSearchSocial.textContent = "⏳ Recherche...";
        try {
            const res = await fetch(`/api/osint/dossier/${encodeURIComponent(name)}`);
            const dossier = await res.json();
            const platforms = dossier.social_platforms || {};
            socialResults.style.display = "grid";
            socialResults.innerHTML = "";
            for (const [key, p] of Object.entries(platforms)) {
                const a = document.createElement("a");
                a.href = p.search_url;
                a.target = "_blank";
                a.style.cssText = btnOsintStyle('#1e2740') + 'text-decoration:none; font-size:0.78rem;';
                a.textContent = `${p.icon} ${p.platform}`;
                socialResults.appendChild(a);
            }
            if (dossier.wiki_bio && dossier.wiki_bio.title) {
                const note = document.createElement("p");
                note.style.cssText = "color:#64748b; font-size:0.75rem; grid-column:1/-1; margin:4px 0 0;";
                note.textContent = `📚 Wikipédia : ${dossier.wiki_bio.extract ? dossier.wiki_bio.extract.slice(0, 120) + '...' : dossier.wiki_bio.title}`;
                socialResults.appendChild(note);
            }
        } catch (e) {
            socialResults.style.display = "block";
            socialResults.innerHTML = '<p style="color:#ef4444;">Erreur lors de la recherche.</p>';
        }
        btnSearchSocial.textContent = "🔎 Chercher";
    });

    // Enrôlement biométrique
    document.getElementById("btnOsintEnroll").addEventListener("click", async () => {
        const name = document.getElementById("osintEnrollName").value.trim();
        const role = document.getElementById("osintEnrollRole").value.trim();
        if (!name) {
            document.getElementById("osintEnrollName").focus();
            showToast("Entrez le nom de la personne.", "error");
            return;
        }
        if (!state.osintLastImagePath) {
            showToast("Aucune image cible disponible.", "error");
            return;
        }
        const btn = document.getElementById("btnOsintEnroll");
        btn.textContent = "⏳ Enrôlement en cours...";
        btn.disabled = true;
        try {
            const res = await fetch("/api/osint/enroll", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, role, notes: "", image_path: state.osintLastImagePath }),
            });
            const data = await res.json();
            const statusDiv = document.getElementById("osintEnrollStatus");
            if (res.ok) {
                statusDiv.style.display = "block";
                statusDiv.style.color = "#10b981";
                statusDiv.textContent = `✅ ${data.message}`;
                showToast(`🎯 ${name} mémorisé ! La caméra le reconnaît maintenant.`, "success");
                loadFaces();
            } else {
                statusDiv.style.display = "block";
                statusDiv.style.color = "#ef4444";
                statusDiv.textContent = `❌ ${data.detail}`;
            }
        } catch (e) {
            showToast("Erreur d'enrôlement.", "error");
        }
        btn.textContent = "💾 Enregistrer dans la caméra — Reconnaissance immédiate";
        btn.disabled = false;
    });
}

function btnOsintStyle(bg) {
    return `display:block; background:${bg}22; border:1px solid ${bg}55; border-radius:9px;
        color:#cbd5e1; padding:9px 14px; text-decoration:none; font-size:0.82rem;
        cursor:pointer; transition: all 0.2s; font-weight:500;`;
}

// Branchement du bouton 🌐 sur le flux vidéo
if (elements.btnWebSearchFace) {
    elements.btnWebSearchFace.addEventListener("click", async () => {
        if (!state.currentAnnotatedImage) {
            showToast("Aucun flux actif. Démarrez la caméra.", "error");
            return;
        }
        elements.btnWebSearchFace.textContent = "⏳";
        try {
            const res = await fetch("/api/osint/target", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: state.currentAnnotatedImage }),
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Erreur serveur");
            }
            const osintData = await res.json();
            openOsintModal(osintData);
            showToast("🔍 Enquête OSINT ouverte !", "success");
        } catch (e) {
            showToast(`Erreur OSINT : ${e.message}`, "error");
        }
        elements.btnWebSearchFace.textContent = "🌐";
    });
}

// ==============================================================================
// Gestion de la Source Vidéo (PC vs Téléphone)
// ==============================================================================

elements.radioSources.forEach((radio) => {
    radio.addEventListener("change", async (e) => {
        state.mode = e.target.value;
        if (state.mode === "browser") {
            elements.pcCameraIndexGroup.style.display = "none";
            await startBrowserCamera();
        } else {
            stopBrowserCamera();
            elements.pcCameraIndexGroup.style.display = "block";
            initWebSocketStream();
        }
    });
});

elements.btnApplyCamIndex.addEventListener("click", async () => {
    const idx = parseInt(elements.pcCamIndexInput.value, 10);
    await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pc_camera_index: idx }),
    });
    showToast(`Index webcam PC réglé sur #${idx}`);
    if (state.mode === "pc") {
        initWebSocketStream();
    }
});

// ==============================================================================
// Configuration YOLO & Mode Vocabulaire Ouvert (YOLO-World)
// ==============================================================================

elements.selectModel.addEventListener("change", async (e) => {
    const model = e.target.value;
    await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: model }),
    });
    elements.activeModelBadge.textContent = model;
    showToast(`Modèle YOLOv8 activé : ${model}`);
});

elements.rangeConfidence.addEventListener("input", (e) => {
    elements.confVal.textContent = `${Math.round(e.target.value * 100)}%`;
});

elements.rangeConfidence.addEventListener("change", async (e) => {
    await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confidence_threshold: parseFloat(e.target.value) }),
    });
});

elements.checkFacesEnabled.addEventListener("change", async (e) => {
    await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ faces_enabled: e.target.checked }),
    });
    showToast(`Reconnaissance faciale : ${e.target.checked ? "Activée" : "Désactivée"}`);
});

// Toggle YOLO-World
elements.btnApplyOpenVocab.addEventListener("click", async () => {
    const rawText = elements.customClassesInput.value.trim();
    const classes = rawText.split(",").map((c) => c.trim()).filter(Boolean);
    const enabled = elements.toggleOpenVocab.checked;

    await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            open_vocabulary_enabled: enabled,
            custom_classes: classes,
        }),
    });

    elements.activeModelBadge.textContent = enabled ? "YOLO-World (Ouvert)" : elements.selectModel.value;
    showToast(enabled ? "Mode 'Tout Reconnaître' activé !" : "Mode YOLO standard activé.");
});

elements.toggleOpenVocab.addEventListener("change", () => {
    elements.btnApplyOpenVocab.click();
});

// ==============================================================================
// Reconnaissance Faciale & Conformité RGPD
// ==============================================================================

async function loadFaces() {
    try {
        const res = await fetch("/api/faces");
        const data = await res.json();
        elements.facesCountBadge.textContent = data.count || 0;
        elements.facesGallery.innerHTML = "";

        if (!data.persons || data.persons.length === 0) {
            elements.facesGallery.innerHTML = '<div class="empty-state">Aucun visage mémorisé pour le moment.</div>';
            return;
        }

        data.persons.forEach((person) => {
            const card = document.createElement("div");
            card.className = "face-card";
            const bioDesc = person.bio && person.bio.description ? `<span class="face-bio-tag" title="${person.bio.description}">${person.bio.description}</span>` : "";
            card.innerHTML = `
                <img class="face-thumb" src="${person.thumbnail_url || ''}" alt="${person.name}" onerror="this.src='/static/img/avatar.svg'">
                <div class="face-info">
                    <div class="face-meta">
                        <span class="face-name" title="${person.name}">${person.name}</span>
                        ${bioDesc}
                    </div>
                    <button class="btn-del-face" data-name="${person.name}" title="Supprimer définitivement">🗑️</button>
                </div>
            `;
            elements.facesGallery.appendChild(card);
        });

        // Écouteurs de suppression
        document.querySelectorAll(".btn-del-face").forEach((btn) => {
            btn.addEventListener("click", async (e) => {
                const name = e.target.getAttribute("data-name");
                if (confirm(`Confirmez-vous la suppression des données biométriques de '${name}' ?`)) {
                    await fetch(`/api/faces/${encodeURIComponent(name)}`, { method: "DELETE" });
                    showToast(`Identité '${name}' supprimée.`);
                    loadFaces();
                }
            });
        });
    } catch (e) {
        console.error("Erreur chargement visages:", e);
    }
}

// Apprentissage automatique depuis Wikipédia / Web
if (elements.btnLearnWebPerson) {
    elements.btnLearnWebPerson.addEventListener("click", async () => {
        const name = elements.webPersonName.value.trim();
        if (!name) {
            showToast("Veuillez saisir le nom d'une personnalité.", "error");
            elements.webPersonName.focus();
            return;
        }

        elements.btnLearnWebPerson.disabled = true;
        elements.btnLearnWebPerson.textContent = "⏳ Recherche Web...";

        try {
            const res = await fetch("/api/web/learn", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name }),
            });
            const data = await res.json();

            if (res.ok) {
                showToast(data.message, "success");
                elements.webPersonName.value = "";
                // Affichage aperçu bio
                if (data.info && elements.webBioPreview) {
                    elements.webBioThumb.src = data.info.image_url || "";
                    elements.webBioTitle.textContent = data.info.name;
                    elements.webBioDesc.textContent = data.info.description || "";
                    elements.webBioExtract.textContent = data.info.extract || "";
                    elements.webBioLink.href = data.info.url || "#";
                    elements.webBioPreview.style.display = "flex";
                }
                loadFaces();
            } else {
                showToast(data.detail || "Échec de l'apprentissage Web.", "error");
            }
        } catch (e) {
            showToast("Erreur lors de la communication serveur.", "error");
        } finally {
            elements.btnLearnWebPerson.disabled = false;
            elements.btnLearnWebPerson.textContent = "⚡ Apprendre du Web";
        }
    });

    elements.webPersonName.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            elements.btnLearnWebPerson.click();
        }
    });
}

// ==============================================================================
// Investigation OSINT & Profiling Réseaux Sociaux
// ==============================================================================

let currentOsintTargetImagePath = "";

const osintElements = {
    modal: document.getElementById("osintModal"),
    backdrop: document.getElementById("osintModalBackdrop"),
    btnClose: document.getElementById("btnCloseOsintModal"),
    targetImg: document.getElementById("osintTargetImg"),
    linkGoogleLens: document.getElementById("linkGoogleLens"),
    linkYandex: document.getElementById("linkYandex"),
    linkFaceCheck: document.getElementById("linkFaceCheck"),
    linkBingVisual: document.getElementById("linkBingVisual"),
    inputFoundName: document.getElementById("inputOsintFoundName"),
    btnSearchSocial: document.getElementById("btnSearchSocialProfiles"),
    socialContainer: document.getElementById("socialLinksContainer"),
    inputRole: document.getElementById("inputOsintRole"),
    inputNotes: document.getElementById("inputOsintNotes"),
    btnEnroll: document.getElementById("btnEnrollInvestigated"),
};

function closeOsintModal() {
    if (osintElements.modal) osintElements.modal.style.display = "none";
}

if (osintElements.btnClose) osintElements.btnClose.addEventListener("click", closeOsintModal);
if (osintElements.backdrop) osintElements.backdrop.addEventListener("click", closeOsintModal);

// Déclenchement de l'Investigation OSINT depuis le flux webcam
if (elements.btnWebSearchFace) {
    elements.btnWebSearchFace.addEventListener("click", async () => {
        let b64 = "";
        if (state.mode === "browser") {
            const ctx = elements.captureCanvas.getContext("2d");
            ctx.drawImage(elements.webcamVideo, 0, 0, 640, 480);
            b64 = elements.captureCanvas.toDataURL("image/jpeg", 0.90);
        } else {
            b64 = state.currentAnnotatedImage;
        }

        if (!b64) {
            showToast("Aucun flux vidéo disponible pour l'investigation.", "error");
            return;
        }

        showToast("🕵️ Préparation de la cible OSINT...", "success");

        try {
            const res = await fetch("/api/osint/target", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: b64 }),
            });
            const data = await res.json();
            if (res.ok) {
                currentOsintTargetImagePath = data.file_path;
                osintElements.targetImg.src = data.image_url;
                osintElements.modal.style.display = "flex";
                showToast("Dossier d'investigation prêt !", "success");
            } else {
                showToast(data.detail || "Échec de préparation OSINT.", "error");
            }
        } catch (e) {
            showToast("Erreur communication serveur OSINT.", "error");
        }
    });
}

// Recherche & Profiling Réseaux Sociaux
if (osintElements.btnSearchSocial) {
    osintElements.btnSearchSocial.addEventListener("click", async () => {
        const name = osintElements.inputFoundName.value.trim();
        if (!name) {
            showToast("Veuillez saisir le nom découvert.", "error");
            osintElements.inputFoundName.focus();
            return;
        }

        osintElements.btnSearchSocial.disabled = true;
        osintElements.btnSearchSocial.textContent = "⏳ Recherche...";

        try {
            const res = await fetch(`/api/osint/dossier/${encodeURIComponent(name)}`);
            const data = await res.json();
            if (res.ok && data.social_platforms) {
                osintElements.socialContainer.innerHTML = "";
                for (const [key, plat] of Object.entries(data.social_platforms)) {
                    const link = document.createElement("a");
                    link.className = "social-chip-link";
                    link.href = plat.search_url;
                    link.target = "_blank";
                    link.innerHTML = `<span>${plat.icon}</span> <span>${plat.platform} ↗</span>`;
                    osintElements.socialContainer.appendChild(link);
                }
                osintElements.socialContainer.style.display = "grid";

                // Auto-remplissage rôle si bio trouvée
                if (data.wiki_bio && data.wiki_bio.description) {
                    osintElements.inputRole.value = data.wiki_bio.description;
                }
                showToast("Profils réseaux sociaux générés !", "success");
            }
        } catch (e) {
            showToast("Erreur profiling réseaux sociaux.", "error");
        } finally {
            osintElements.btnSearchSocial.disabled = false;
            osintElements.btnSearchSocial.textContent = "🔎 Découvrir Profils Sociaux";
        }
    });

    osintElements.inputFoundName.addEventListener("keydown", (e) => {
        if (e.key === "Enter") osintElements.btnSearchSocial.click();
    });
}

// Enrôlement officiel de la personne trouvée
if (osintElements.btnEnroll) {
    osintElements.btnEnroll.addEventListener("click", async () => {
        const name = osintElements.inputFoundName.value.trim();
        const role = osintElements.inputRole.value.trim();
        const notes = osintElements.inputNotes.value.trim();

        if (!name) {
            showToast("Veuillez saisir un nom pour l'enregistrement.", "error");
            osintElements.inputFoundName.focus();
            return;
        }

        osintElements.btnEnroll.disabled = true;
        osintElements.btnEnroll.textContent = "⏳ Enregistrement dans la caméra...";

        try {
            const res = await fetch("/api/osint/enroll", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name,
                    role,
                    notes,
                    image_path: currentOsintTargetImagePath,
                }),
            });
            const data = await res.json();
            if (res.ok) {
                showToast(data.message, "success");
                closeOsintModal();
                loadFaces();
            } else {
                showToast(data.detail || "Échec de l'enregistrement.", "error");
            }
        } catch (e) {
            showToast("Erreur serveur.", "error");
        } finally {
            osintElements.btnEnroll.disabled = false;
            osintElements.btnEnroll.textContent = "💾 Enregistrer et Faire Reconnaître en Direct par la Caméra";
        }
    });
}

// Enregistrement par capture instantanée du flux
elements.btnCaptureCurrentFace.addEventListener("click", async () => {
    const name = elements.facePersonName.value.trim();
    if (!name) {
        showToast("Veuillez saisir un nom ou prénom.", "error");
        elements.facePersonName.focus();
        return;
    }

    let b64 = "";
    if (state.mode === "browser") {
        const ctx = elements.captureCanvas.getContext("2d");
        ctx.drawImage(elements.webcamVideo, 0, 0, 640, 480);
        b64 = elements.captureCanvas.toDataURL("image/jpeg", 0.90);
    } else {
        b64 = state.currentAnnotatedImage;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("image_base64", b64);

    try {
        const res = await fetch("/api/faces/register", { method: "POST", body: formData });
        const data = await res.json();
        if (res.ok) {
            showToast(data.message, "success");
            elements.facePersonName.value = "";
            loadFaces();
        } else {
            showToast(data.detail || "Échec de l'enregistrement.", "error");
        }
    } catch (e) {
        showToast("Erreur communication serveur.", "error");
    }
});

// Enregistrement par téléversement de fichier
elements.faceFileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    const name = elements.facePersonName.value.trim();
    if (!file) return;
    if (!name) {
        showToast("Veuillez d'abord saisir un nom avant de choisir un fichier.", "error");
        return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("file", file);

    try {
        const res = await fetch("/api/faces/register", { method: "POST", body: formData });
        const data = await res.json();
        if (res.ok) {
            showToast(data.message, "success");
            elements.facePersonName.value = "";
            elements.faceFileInput.value = "";
            loadFaces();
        } else {
            showToast(data.detail || "Échec de l'enregistrement.", "error");
        }
    } catch (e) {
        showToast("Erreur serveur.", "error");
    }
});

// Purge totale des visages
elements.btnPurgeFaces.addEventListener("click", async () => {
    if (confirm("ATTENTION : Voulez-vous supprimer DÉFINITIVEMENT tous les visages enregistrés ?")) {
        await fetch("/api/faces", { method: "DELETE" });
        showToast("Toutes les données faciales ont été supprimées.", "success");
        loadFaces();
    }
});

// ==============================================================================
// Galerie de Captures Sauvegardées
// ==============================================================================

async function loadCaptures() {
    try {
        const res = await fetch("/api/captures");
        const data = await res.json();
        elements.capturesGallery.innerHTML = "";

        if (!data.captures || data.captures.length === 0) {
            elements.capturesGallery.innerHTML = '<div class="empty-state">Aucune capture enregistrée.</div>';
            return;
        }

        data.captures.forEach((cap) => {
            const card = document.createElement("div");
            card.className = "capture-card";
            card.innerHTML = `
                <a href="${cap.url}" target="_blank" title="Ouvrir en plein écran">
                    <img class="capture-thumb" src="${cap.url}" alt="${cap.filename}">
                </a>
                <div class="capture-info">
                    <span style="font-size:0.7rem; color:var(--text-muted);">${cap.timestamp}</span>
                    <a href="${cap.url}" download="${cap.filename}" style="color:var(--accent-cyan); text-decoration:none;">⬇️</a>
                </div>
            `;
            elements.capturesGallery.appendChild(card);
        });
    } catch (e) {
        console.error("Erreur captures:", e);
    }
}

elements.btnRefreshCaptures.addEventListener("click", loadCaptures);

// ==============================================================================
// Gestion des Onglets
// ==============================================================================

document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));

        btn.classList.add("active");
        const target = document.getElementById(btn.getAttribute("data-tab"));
        if (target) target.classList.add("active");

        if (btn.getAttribute("data-tab") === "tab-faces") loadFaces();
        if (btn.getAttribute("data-tab") === "tab-gallery") loadCaptures();
        if (btn.getAttribute("data-tab") === "tab-enterprise") {
            loadEnterpriseConfig();
            loadAuditLogs();
        }
    });
});

// ==============================================================================
// Sûreté Entreprise & Audit Log
// ==============================================================================

async function loadEnterpriseConfig() {
    try {
        const res = await fetch("/api/enterprise/config");
        if (!res.ok) return;
        const data = await res.json();
        if (elements.checkRestrictedZone) elements.checkRestrictedZone.checked = !!data.restricted_zone_enabled;
        if (elements.checkFallDetection) elements.checkFallDetection.checked = !!data.fall_detection_enabled;
        if (elements.inputLoitering) elements.inputLoitering.value = data.loitering_threshold_s || 6;
    } catch (e) {
        console.error("Erreur config entreprise:", e);
    }
}

async function loadAuditLogs() {
    if (!elements.auditTableBody) return;
    try {
        const res = await fetch("/api/enterprise/audit-logs?limit=40");
        if (!res.ok) return;
        const data = await res.json();
        elements.auditTableBody.innerHTML = "";
        if (!data.logs || data.logs.length === 0) {
            elements.auditTableBody.innerHTML = '<tr><td colspan="6" class="empty-state">Aucun incident de sûreté enregistré.</td></tr>';
            return;
        }
        data.logs.forEach((log) => {
            const tr = document.createElement("tr");
            const sevClass = log.severity === "CRITICAL" ? "badge-critical" : (log.severity === "WARNING" ? "badge-warning" : "badge-info");
            const proofHtml = log.snapshot_path ? `<a href="${log.snapshot_path}" target="_blank" class="audit-thumb-link">📸 Voir</a>` : "-";
            tr.innerHTML = `
                <td><code>${log.timestamp}</code></td>
                <td><span class="badge-severity ${sevClass}">${log.severity}</span></td>
                <td><strong>${log.event_type}</strong></td>
                <td>${log.person_name}</td>
                <td>${log.description}</td>
                <td>${proofHtml}</td>
            `;
            elements.auditTableBody.appendChild(tr);
        });
    } catch (e) {
        console.error("Erreur audit logs:", e);
    }
}

if (elements.btnSaveBehaviorConfig) {
    elements.btnSaveBehaviorConfig.addEventListener("click", async () => {
        const payload = {
            restricted_zone_enabled: elements.checkRestrictedZone.checked,
            fall_detection_enabled: elements.checkFallDetection.checked,
            loitering_threshold_s: parseFloat(elements.inputLoitering.value) || 6.0,
        };
        try {
            const res = await fetch("/api/enterprise/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (res.ok) {
                showToast("Règles de sûreté mises à jour avec succès !", "success");
            }
        } catch (e) {
            showToast("Erreur mise à jour règles sûreté.", "error");
        }
    });

    elements.checkRestrictedZone.addEventListener("change", () => elements.btnSaveBehaviorConfig.click());
    elements.checkFallDetection.addEventListener("change", () => elements.btnSaveBehaviorConfig.click());
}

if (elements.btnConnectRtsp) {
    elements.btnConnectRtsp.addEventListener("click", async () => {
        const url = elements.inputRtspUrl.value.trim();
        if (!url) {
            showToast("Veuillez renseigner une URL RTSP valide.", "error");
            return;
        }
        elements.btnConnectRtsp.disabled = true;
        elements.btnConnectRtsp.textContent = "Connexion...";
        try {
            const res = await fetch("/api/enterprise/rtsp", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url }),
            });
            const data = await res.json();
            if (res.ok) {
                showToast(data.message, "success");
                const pcRadio = document.querySelector("input[name='cameraSource'][value='pc']");
                if (pcRadio) {
                    pcRadio.checked = true;
                    pcRadio.dispatchEvent(new Event("change"));
                }
            } else {
                showToast(data.detail || "Échec connexion caméra IP.", "error");
            }
        } catch (e) {
            showToast("Erreur communication serveur.", "error");
        } finally {
            elements.btnConnectRtsp.disabled = false;
            elements.btnConnectRtsp.textContent = "Connecter IP";
        }
    });
}

if (elements.btnRefreshAudit) elements.btnRefreshAudit.addEventListener("click", loadAuditLogs);

// ==============================================================================
// Contrôle d'Accès Physique & Anti-Spoofing
// ==============================================================================

async function loadActionsStatus() {
    try {
        const res = await fetch("/api/enterprise/actions/status");
        if (!res.ok) return;
        const data = await res.json();
        if (elements.checkLiveness) elements.checkLiveness.checked = !!data.liveness_enabled;
        if (elements.inputWebhookUrl && data.status && data.status.webhook_url) {
            elements.inputWebhookUrl.value = data.status.webhook_url;
        }
        if (elements.doorRelayStatusBadge && data.status) {
            const r = data.status.door_relay;
            elements.doorRelayStatusBadge.textContent = `Statut : ${r === "UNLOCKED" ? "OUVERT" : (r === "LOCKDOWN" ? "CONFINEMENT" : "VERROUILLÉ")}`;
            elements.doorRelayStatusBadge.className = `badge-severity ${r === "UNLOCKED" ? "badge-info" : (r === "LOCKDOWN" ? "badge-critical" : "badge-warning")}`;
        }
    } catch (e) {
        console.error("Erreur actions status:", e);
    }
}

if (elements.btnDoorUnlock) {
    elements.btnDoorUnlock.addEventListener("click", async () => {
        try {
            const res = await fetch("/api/enterprise/actions/door", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "UNLOCK", duration_s: 5.0 }),
            });
            if (res.ok) {
                showToast("Porte déverrouillée (5 secondes).", "success");
                loadActionsStatus();
                setTimeout(loadActionsStatus, 5200);
            }
        } catch (e) {
            showToast("Erreur commande porte.", "error");
        }
    });
}

if (elements.btnDoorLockdown) {
    elements.btnDoorLockdown.addEventListener("click", async () => {
        if (confirm("URGENCE : Déclencher le confinement immédiat du périmètre ?")) {
            try {
                const res = await fetch("/api/enterprise/actions/door", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ action: "LOCKDOWN" }),
                });
                if (res.ok) {
                    showToast("🚨 CONFINEMENT ACTIVÉ (Portes verrouillées).", "error");
                    loadActionsStatus();
                }
            } catch (e) {
                showToast("Erreur commande confinement.", "error");
            }
        }
    });
}

if (elements.btnSaveWebhook) {
    elements.btnSaveWebhook.addEventListener("click", async () => {
        const url = elements.inputWebhookUrl.value.trim();
        try {
            const res = await fetch("/api/enterprise/actions/webhook", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url }),
            });
            if (res.ok) {
                showToast("Webhook d'urgence enregistré avec succès !", "success");
            }
        } catch (e) {
            showToast("Erreur enregistrement webhook.", "error");
        }
    });
}

if (elements.checkLiveness) {
    elements.checkLiveness.addEventListener("change", async () => {
        try {
            const res = await fetch("/api/enterprise/liveness", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ enabled: elements.checkLiveness.checked }),
            });
            if (res.ok) {
                const st = elements.checkLiveness.checked ? "activé" : "désactivé";
                showToast(`Anti-Spoofing (Détection de vivant) ${st}.`, "success");
            }
        } catch (e) {
            showToast("Erreur mise à jour anti-spoofing.", "error");
        }
    });
}

// ==============================================================================
// Démarrage au Chargement de la Page
// ==============================================================================

window.addEventListener("DOMContentLoaded", async () => {
    if (elements.deviceSubtitle) {
        elements.deviceSubtitle.textContent = state.isMobile
            ? "Mode Mobile (Caméra Smartphone)"
            : "Mode Desktop (Webcam PC)";
    }

    await loadServerConfig();
    loadFaces();
    loadCaptures();
    loadEnterpriseConfig();
    loadAuditLogs();
    loadActionsStatus();

    // Démarrage automatique sur la caméra du navigateur / téléphone
    if (state.mode === "browser") {
        await startBrowserCamera();
    } else {
        initWebSocketStream();
    }

    // Gestion de la clé API d'entreprise dans l'interface
    const inputEnterpriseApiKey = document.getElementById("inputEnterpriseApiKey");
    const btnSaveApiKey = document.getElementById("btnSaveApiKey");
    if (inputEnterpriseApiKey && btnSaveApiKey) {
        inputEnterpriseApiKey.value = localStorage.getItem("enterprise_api_key") || "";
        btnSaveApiKey.addEventListener("click", () => {
            const key = inputEnterpriseApiKey.value.trim();
            if (key) {
                localStorage.setItem("enterprise_api_key", key);
                showToast("Clé d'API d'entreprise sauvegardée dans le navigateur !", "success");
            } else {
                localStorage.removeItem("enterprise_api_key");
                showToast("Clé d'API retirée du navigateur.", "info");
            }
        });
    }
});

