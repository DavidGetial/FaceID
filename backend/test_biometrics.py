import numpy as np
from deepface import DeepFace

print("Iniciando prueba del motor biométrico...")

# Creamos una imagen sintética de prueba (224x224 píxeles)
dummy_image = np.ones((224, 224, 3), dtype=np.uint8) * 255

try:
    # Extraemos el vector facial omitiendo el detector predeterminado de OpenCV
    embedding_objs = DeepFace.represent(
        img_path=dummy_image,
        model_name="SFace",
        enforce_detection=False,
        detector_backend="skip"  # <-- Evita buscar el archivo XML de OpenCV
    )

    vector = embedding_objs[0]["embedding"]
    print("\n¡Éxito! El motor biométrico está funcionando perfectamente.")
    print(f"Modelo cargado: SFace")
    print(f"Dimensión del vector facial: {len(vector)} números")
    print(f"Muestra de los primeros 5 valores: {vector[:5]}")

except Exception as e:
    print(f"\nError al probar el motor: {e}")