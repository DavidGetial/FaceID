# 🧠 FaceID Pro Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![DeepFace](https://img.shields.io/badge/DeepFace-SFace-FF6F00)](https://github.com/serengil/deepface)
[![License](https://img.shields.io/badge/Uso-Educativo%20%2F%20Investigaci%C3%B3n-blue)](#licencia-y-consideraciones)

**FaceID Pro Engine** es un sistema biométrico *end-to-end* de **reconocimiento facial y control de acceso**. Combina una interfaz web moderna con un backend de alto rendimiento para detectar, verificar y registrar rostros en tiempo real, con comprobaciones de calidad de imagen y *liveness* básico anti suplantación por foto estática.

> Desarrollado como proyecto de ingeniería de software con foco en arquitectura limpia, experiencia de usuario y despliegue reproducible.

---

## ✨ Características principales

- **Detección facial en tiempo real** con YuNet/mtcnn sobre la cámara del navegador.
- **Embeddings faciales SFace** (ligero y portable, ideal para ejecución local o en edge).
- **Verificación por similitud del coseno** entre el rostro capturado y los perfiles registrados.
- **Liveness simple**: exige variación entre frames capturados consecutivos (parpadeo/movimiento).
- **Control de calidad de imagen**: brillo, contraste y viabilidad del rostro antes de extraer el embedding.
- **Enrolamiento incremental**: hasta 200 muestras por usuario con actualización automática del embedding maestro (promedio normalizado).
- **Gestión de usuarios**: listado, enrolamiento y eliminación.
- **Historial de accesos** persistente con exportación a CSV (usuarios e historial).
- **Interfaz web de una sola página** (HTML/CSS/JS) con guías visuales, estados y feedback en tiempo real.
- **API REST documentada** automáticamente por FastAPI (`/docs`).

---

## 🏗️ Arquitectura

```text
faceid_engine/
├── backend/
│   ├── main.py              # API FastAPI + lógica de reconocimiento
│   ├── database.py          # Persistencia JSON + similitud del coseno
│   ├── prof_model/          # Modelos auxiliares (SFace / Teachable Machine)
│   └── test_*.py            # Pruebas manuales de BD y biometría
├── frontend/
│   └── index.html           # Interfaz web (SPA)
├── database/
│   ├── users.json           # Perfiles y embeddings normalizados
│   ├── access_log.json      # Historial de accesos
│   ├── samples/             # Imágenes de enrolamiento por usuario
│   └── *_backup*.json       # Respaldos
├── requirements.txt
├── .gitignore
└── README.md
```

### Flujo de una verificación

1. El navegador captura varios frames consecutivos y los envía a `/api/recognize`.
2. Se aplica la comprobación de *liveness* (> umbral de variación entre frames).
3. Cada frame se decodifica y se valida calidad (brillo/contraste).
4. Se detecta el rostro (YuNet, respaldo mtcnn) y se extrae el embedding **SFace**.
5. Se promedian los embeddings de los frames capturados y se normaliza a vector unitario.
6. Se compara contra cada muestra almacenada de cada usuario (similitud del coseno).
7. Se decide: `GRANTED (≥ 60%)`, `UNCERTAIN (≥ 45%)` o `DENIED`.
8. Se registra el evento en `access_log.json` y se devuelve el resultado con confianza.

---

## 🧰 Stack tecnológico

| Capa | Tecnología |
|---|---|
| Backend | **FastAPI** + **Uvicorn** + **Pydantic** |
| Visión por computador | **OpenCV** (detección YuNet y calidad de imagen) |
| Embeddings faciales | **DeepFace** con modelo **SFace** |
| Detección (respaldo) | **mtcnn** |
| Álgebra vectorial | **NumPy** (similitud del coseno, normalización) |
| Persistencia | Archivos **JSON** + imágenes en disco |
| Frontend | **HTML5, CSS3, JavaScript** (SPA sin bundler) |

---

## 🚀 Instalación y ejecución

### 1. Requisitos previos

- **Python 3.10 o superior**
- Webcam disponible y permisos de navegador habilitados
- (Opcional) Git para clonar el repositorio

### 2. Clonar el repositorio

```bash
git clone https://github.com/DavidGetial/FaceID.git
cd FaceID
```

### 3. Crear y activar un entorno virtual

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Ejecutar el servidor

```bash
python backend/main.py
```

El servidor queda disponible en:

- Aplicación web: http://localhost:8000/
- Documentación API (Swagger): http://localhost:8000/docs

> 💡 En el primer arranque DeepFace descarga y calienta el modelo SFace; puede tardar unos segundos.

---

## 🖥️ Uso paso a paso

### Registrar un usuario

1. Abre http://localhost:8000/ y ve a la pestaña **Administración**.
2. Escribe el **nombre** y el **rol**.
3. Captura una o varias fotos del rostro (mejor con luz suave, sin gafas y mirando de frente).
4. Cada muestra se guarda y el perfil actualiza su embedding maestro.

### Verificar acceso

1. En la pestaña **Verificación**, mira a la cámara.
2. El sistema muestra un anillo de progreso y el estado: `ACCESO`, `DUDA` o `NO RECONOCIDO`.
3. El evento queda registrado en el historial.

### Administrar

- **Usuarios**: listar y eliminar perfiles.
- **Historial**: revisar accesos recientes.
- **Exportar**: descargar CSV de usuarios o historial.

---

## 🔌 API REST

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Sirve la interfaz web |
| `POST` | `/api/face-check` | Detecta un rostro y devuelve bounding box |
| `POST` | `/api/recognize` | Verifica identidad a partir de varios frames |
| `GET` | `/api/users` | Lista usuarios registrados |
| `POST` | `/api/enroll` | Registra/actualiza un usuario con una muestra |
| `DELETE` | `/api/users/{user_id}` | Elimina un usuario |
| `GET` | `/api/history` | Historial de accesos |
| `DELETE` | `/api/history` | Limpia el historial |
| `GET` | `/api/export/users` | Exporta usuarios a CSV |
| `GET` | `/api/export/history` | Exporta historial a CSV |

---

## 📦 Formato de datos

### `database/users.json`

```json
[
  {
    "id": "1",
    "name": "david",
    "role": "admin",
    "embedding": [0.0374, -0.0124, "..."],
    "samples": [["..."], ["..."]],
    "photos": ["/photos/1/sample_1.jpg"]
  }
]
```

### `database/access_log.json`

```json
[
  {
    "time": "2026-10-14T22:48:13",
    "access": "GRANTED",
    "confidence": 82.5,
    "user": "david",
    "tip": "Acceso verificado."
  }
]
```

---

## ⚙️ Configuración y parámetros clave

- **Puerto del servidor**: `8000` (modificable en `backend/main.py` al invocar `uvicorn`).
- **Directorio de muestras**: `database/samples/<user_id>/`.
- **Límite de muestras por usuario**: `200`.
- **Límite del historial**: `1000` eventos.
- **Umbral de acceso**: `60%` de confianza.
- **Umbral de duda**: `45%` de confianza.
- **Tamaño máximo de detección**: `360px` por lado (optimización de rendimiento).
- **Liveness**: diferencia media mínima de `2.5` entre frames de `64x64`.

Modifica estos valores directamente en `backend/main.py` si el caso de uso lo exige.

---

## 🛡️ Privacidad y seguridad

Este proyecto se orienta a aprendizaje/investigación. Antes de usarlo en producción considera:

- Añadir autenticación (JWT/OAuth2) a los endpoints de administración.
- Restringir CORS y exponer solo los endpoints necesarios.
- Encriptar o anonimizar plantillas biométricas en reposo.
- Cumplir con la normativa local de protección de datos biométricos (RGPD/CCPA).
- Excluir `database/samples/` y `users.json` del control de versiones si contienen datos personales.

---

## 🧪 Pruebas manuales

```bash
python backend/test_db.py
python backend/test_biometrics.py
```

*(Scripts auxiliares para validar la base de datos y el pipeline biométrico.)*

---

## 🛠️ Roadmap sugerido

- [ ] Autenticación de administradores
- [ ] Reemplazo de JSON por base de datos relacional (PostgreSQL)
- [ ] Detección de anti-spoofing más robusta (parpadeo + profundidad)
- [ ] Exportación de métricas y dashboard
- [ ] Empaquetado con Docker
- [ ] Tests unitarios (pytest)

---

## 👤 Autor

**David Getial**  
GitHub: [@DavidGetial](https://github.com/DavidGetial)

---

## 📄 Licencia y consideraciones

Proyecto con fines educativos y de investigación. Si reutilizas el código, respeta la normativa de protección de datos y las licencias de las dependencias empleadas (FastAPI, OpenCV, DeepFace, etc.).
