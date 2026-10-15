import numpy as np
try:
    from backend.database import find_matching_user, save_user
except ModuleNotFoundError:
    from database import find_matching_user, save_user

print("=== PROBANDO BASE DE DATOS BIOMÉTRICA ===")

# 1. Simulamos el vector facial de "David (Profesor)"
vector_david = np.random.rand(512).tolist()

# 2. Registramos a David en la base de datos
save_user(
    user_id="USR-001",
    name="Profesor David",
    role="Docente / Administrador",
    embedding=vector_david,
)

# 3. Prueba A: Simulamos que la cámara lee exactamente la misma cara
user_found, match_score = find_matching_user(vector_david)
print(f"\nPrueba A (Misma persona):")
print(
    f"Resultado: {user_found['name'] if user_found else 'Desconocido'} | Coincidencia: {match_score}%"
)

# 4. Prueba B: Simulamos un rostro desconocido completamente diferente
vector_desconocido = np.random.rand(512).tolist()
user_found_b, match_score_b = find_matching_user(vector_desconocido)
print(f"\nPrueba B (Desconocido/Intruso):")
print(
    f"Resultado: {user_found_b['name'] if user_found_b else 'Desconocido / Acceso Denegado'} | Coincidencia: {match_score_b}%"
)