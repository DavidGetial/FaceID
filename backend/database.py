import json
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "users.json")

def _ensure_db():
    os.makedirs(DB_DIR, exist_ok=True)
    if not os.path.exists(DB_PATH):
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)

def load_users():
    _ensure_db()
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_user(user_id: str, name: str, role: str, embedding: list):
    users = load_users()
    
    # Si el usuario ya existe por ID, se actualiza
    user_idx = next((i for i, u in enumerate(users) if str(u.get("id")) == str(user_id)), None)
    record = {
        "id": user_id,
        "name": name,
        "role": role,
        "embedding": embedding
    }
    
    if user_idx is not None:
        users[user_idx] = record
    else:
        users.append(record)

    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)
    return record

def cosine_similarity(v1, v2):
    vec1 = np.array(v1, dtype=np.float32)
    vec2 = np.array(v2, dtype=np.float32)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def find_matching_user(embedding_query: list):
    users = load_users()
    if not users:
        return None, 0.0

    best_user = None
    best_similarity = -1.0

    for user in users:
        if "embedding" not in user:
            continue
        sim = cosine_similarity(embedding_query, user["embedding"])
        if sim > best_similarity:
            best_similarity = sim
            best_user = user

    confidence_pct = round(max(0.0, best_similarity) * 100, 2)
    return best_user, confidence_pct