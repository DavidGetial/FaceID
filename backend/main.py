import os
import sys

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import cv2
import numpy as np
import time
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

try:
    from backend.database import load_users, save_user, cosine_similarity, find_matching_user
except ModuleNotFoundError:
    from database import load_users, save_user, cosine_similarity, find_matching_user

app = FastAPI(title="FaceID Pro Engine")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
SAMPLES_DIR = os.path.join(BASE_DIR, "database", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)
LOG_PATH = os.path.join(BASE_DIR, "database", "access_log.json")
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", encoding="utf-8") as _f:
        _f.write("[]")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if os.path.exists(SAMPLES_DIR):
    app.mount("/photos", StaticFiles(directory=SAMPLES_DIR), name="photos")

def detect_face(image_bytes: bytes):
    """Devuelve (embedding, box_norm, det_size, issue) o (None, None, None, issue_str)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None, None, None, "No se detecta imagen. Mira a la camara."

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    if mean < 20.0 or mean > 240.0 or std < 20.0:
        if mean < 60.0:
            return None, None, None, "Ponte en un lugar mas claro."
        if mean > 200.0:
            return None, None, None, "Evita contraluz; busca luz suave."
        return None, None, None, "Acerca tu rostro y mira de frente."

    orig_h, orig_w = frame.shape[:2]
    max_side = max(frame.shape[0], frame.shape[1])
    if max_side > 360:
        scale = 360.0 / max_side
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    det_h, det_w = frame.shape[:2]

    from deepface import DeepFace
    # Modelo ligero estilo movil (OpenCV SFace) + detector YuNet = rapido y portable.
    for backend in ("yunet", "mtcnn"):
        try:
            reps = DeepFace.represent(
                img_path=frame,
                model_name="SFace",
                detector_backend=backend,
                enforce_detection=True
            )
            if reps and len(reps) > 0:
                primary = max(reps, key=lambda r: r.get("facial_area", {}).get("w", 0) * r.get("facial_area", {}).get("h", 0))
                fa = primary["facial_area"]
                box = {
                    "x": fa["x"] / det_w,
                    "y": fa["y"] / det_h,
                    "w": fa["w"] / det_w,
                    "h": fa["h"] / det_h,
                }
                return primary["embedding"], box, (det_w, det_h), None
        except Exception:
            continue
    return None, None, None, "Quitate las gafas y mira de frente, con buena luz."

def extract_face_embedding(image_bytes: bytes):
    emb, _, _, _ = detect_face(image_bytes)
    return emb

def _is_live(contents: list):
    """Liveness simple: exige variacion entre frames (parpadeo/movimiento)."""
    try:
        frames = []
        for c in contents:
            arr = np.frombuffer(c, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return True
            img = cv2.resize(img, (64, 64))
            frames.append(img.astype(np.float32))
        if len(frames) < 2:
            return True
        diffs = []
        for i in range(1, len(frames)):
            diffs.append(float(np.mean(np.abs(frames[i] - frames[i-1]))))
        return max(diffs) >= 2.5
    except Exception:
        return True

def _log_access(result: dict, user):
    try:
        import json as _j
        from datetime import datetime
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as _f:
                data = _j.load(_f)
        except Exception:
            data = []
        if not isinstance(data, list):
            data = []
        entry = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "access": result.get("access"),
            "confidence": result.get("confidence", 0.0),
            "user": (user.get("name") if isinstance(user, dict) else None),
            "tip": result.get("tip"),
        }
        data.append(entry)
        if len(data) > 1000:
            data = data[-1000:]
        with open(LOG_PATH, "w", encoding="utf-8") as _f:
            _j.dump(data, _f, indent=2, ensure_ascii=False)
    except Exception:
        pass

@app.on_event("startup")
def warmup():
    import threading
    def _warm():
        try:
            from deepface import DeepFace
            DeepFace.build_model("SFace")
        except Exception:
            pass
        try:
            from mtcnn import MTCNN
            MTCNN()
        except Exception:
            pass
    threading.Thread(target=_warm, daemon=True).start()

@app.post("/api/face-check")
async def face_check(file: UploadFile = File(...)):
    content = await file.read()
    emb, box, size, issue = detect_face(content)
    if emb is None:
        return {"status": "no_face", "tip": issue}
    return {"status": "face", "box": box}

@app.get("/")
def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "online"}

@app.post("/api/recognize")
async def recognize(files: List[UploadFile] = File(...)):
    contents = [await f.read() for f in files]
    if not _is_live(contents):
        result = {"access": "DENIED", "tip": "Parpadea durante la verificacion para confirmar que eres persona real.", "confidence": 0.0}
        _log_access(result, None)
        return result

    embeddings = []
    first_issue = None
    for content in contents:
        emb = extract_face_embedding(content)
        if emb is not None:
            embeddings.append(emb)
        elif first_issue is None:
            _, _, _, first_issue = detect_face(content)
    if not embeddings:
        result = {"access": "NO_FACE", "tip": first_issue or "No se detecta un rostro. Mira a la camara y quitate las gafas.", "confidence": 0.0}
        _log_access(result, None)
        return result

    arrs = [np.array(e, dtype=np.float32) for e in embeddings]
    q = np.mean(arrs, axis=0)
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n

    users = load_users()
    if not users:
        result = {"access": "DENIED", "tip": "No hay personas registradas. Ve a Administracion.", "confidence": 0.0}
        _log_access(result, None)
        return result

    best_user = None
    best_sim = -1.0
    for u in users:
        samples = u.get("samples") or ([u["embedding"]] if "embedding" in u else [])
        for smp in samples:
            sim = cosine_similarity(q.tolist(), smp)
            if sim > best_sim:
                best_sim = sim
                best_user = u

    if best_user is None:
        result = {"access": "DENIED", "tip": "Sin coincidencias.", "confidence": 0.0}
        _log_access(result, None)
        return result

    confidence = round(max(0.0, best_sim) * 100, 2)

    if confidence >= 60.0:
        result = {
            "access": "GRANTED",
            "user": best_user,
            "confidence": confidence,
            "tip": "Acceso verificado."
        }
        _log_access(result, best_user)
        return result
    elif confidence >= 45.0:
        result = {
            "access": "UNCERTAIN",
            "confidence": confidence,
            "tip": f"Duda ({confidence}%). Quitate las gafas, acerca tu rostro y mira de frente."
        }
        _log_access(result, best_user)
        return result
    else:
        result = {
            "access": "DENIED",
            "confidence": confidence,
            "tip": "No reconocido. Quitate las gafas y registra a la persona o acerca tu rostro."
        }
        _log_access(result, best_user)
        return result

# ---------- Historial y exportacion ----------
@app.get("/api/history")
def get_history(limit: int = 200):
    try:
        import json as _j
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            data = _j.load(f)
    except Exception:
        data = []
    if not isinstance(data, list):
        data = []
    return list(reversed(data))[: max(1, min(limit, 1000))]

@app.delete("/api/history")
def clear_history():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("[]")
    return {"status": "ok"}

@app.get("/api/export/users")
def export_users():
    import csv, io
    users = load_users()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "name", "role", "samples"])
    for u in users:
        w.writerow([u.get("id"), u.get("name"), u.get("role"), len(u.get("samples") or [])])
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=users.csv"})

@app.get("/api/export/history")
def export_history():
    try:
        import json as _j
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            data = _j.load(f)
    except Exception:
        data = []
    if not isinstance(data, list):
        data = []
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "access", "confidence", "user", "tip"])
    for e in data:
        w.writerow([e.get("time"), e.get("access"), e.get("confidence"), e.get("user"), e.get("tip")])
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=history.csv"})

# ---------- Administracion de rostros ----------
@app.get("/api/users")
def list_users():
    users = load_users()
    return [{
        "id": u.get("id"),
        "name": u.get("name"),
        "role": u.get("role"),
        "samples": len(u.get("samples") or []),
        "photos": u.get("photos") or ([f"/photos/{u.get('id')}/sample_1.jpg"] if False else (u.get("photos") or [])),
    } for u in users]

@app.post("/api/enroll")
async def enroll(name: str = Form(...), role: str = Form(""), file: UploadFile = File(...)):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Nombre requerido.")
    content = await file.read()
    emb = extract_face_embedding(content)
    if emb is None:
        raise HTTPException(status_code=400, detail="No se detecta un rostro. Quitate las gafas, ponte en un lugar mas claro y mira a la camara.")

    users = load_users()
    vec = np.array(emb, dtype=np.float32)
    nrm = np.linalg.norm(vec)
    if nrm > 0:
        vec = vec / nrm

    idx = next((i for i, u in enumerate(users) if str(u.get("name", "")).strip().lower() == name.lower()), None)
    if idx is not None:
        user = users[idx]
        samples = user.get("samples") or ([user["embedding"]] if "embedding" in user else [])
        samples.append(vec.tolist())
        if len(samples) > 200:
            samples = samples[-200:]
        arr = np.array(samples, dtype=np.float32)
        master = arr.mean(axis=0)
        mn = np.linalg.norm(master)
        if mn > 0:
            master = master / mn
        user["samples"] = samples
        user["embedding"] = master.tolist()
        user["role"] = role or user.get("role", "Usuario")
        user_id = str(user["id"])
        count = len(samples)
    else:
        user_id = str(len(users) + 1)
        new_user = {
            "id": user_id,
            "name": name,
            "role": role or "Usuario",
            "embedding": vec.tolist(),
            "samples": [vec.tolist()],
            "photos": [],
        }
        users.append(new_user)
        count = 1

    # Guardar imagen de muestra en disco
    user_obj = next(u for u in users if str(u.get("id")) == user_id)
    photos = user_obj.get("photos", [])
    os.makedirs(os.path.join(SAMPLES_DIR, user_id), exist_ok=True)
    photo_name = f"sample_{count}.jpg"
    photo_path = os.path.join(SAMPLES_DIR, user_id, photo_name)
    try:
        with open(photo_path, "wb") as pf:
            pf.write(content)
        photos.append(f"/photos/{user_id}/{photo_name}")
        if len(photos) > 200:
            photos = photos[-200:]
        user_obj["photos"] = photos
    except Exception:
        pass

    import json
    with open(os.path.join(BASE_DIR, "database", "users.json"), "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

    return {"status": "ok", "name": name, "count": count, "message": f"Muestra {count} registrada para {name}."}

@app.delete("/api/users/{user_id}")
def delete_user(user_id: str):
    users = load_users()
    users = [u for u in users if str(u.get("id")) != str(user_id)]
    import json
    with open(os.path.join(BASE_DIR, "database", "users.json"), "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
