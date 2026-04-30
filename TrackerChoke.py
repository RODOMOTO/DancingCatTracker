import cv2
import mediapipe as mp
import math
import numpy as np

# ---------- Configuración ----------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------- Inicialización ----------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def init_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("❌ No se pudo abrir la cámara.")

    # Reducir resolución para mejorar rendimiento
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    return cap

def process_frame(frame, holistic):
    # Convertir BGR → RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    results = holistic.process(image_rgb)

    # Convertir de regreso RGB → BGR
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_bgr, results

def draw_landmarks(image, results):
    # Rostro
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 255, 121), thickness=1, circle_radius=1)
        )

    # Mano izquierda
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # Mano derecha
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    return image

# -------------------  FUNCIONES DE IMAGEN Y VALIDACION  ----------------------

def load_png(image_path, target_width=150):
    """Carga una imagen PNG, mantiene la transparencia y redimensiona."""
    # IMREAD_UNCHANGED es vital para cargar el canal Alfa (RGBA)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"❌ No se encontró el archivo: {image_path}")
        return None

    # Calcular nueva altura manteniendo la proporción
    aspect_ratio = img.shape[0] / img.shape[1]
    target_height = int(target_width * aspect_ratio)
    img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Asegurarnos de que tenga 4 canales (BGRA) por si el PNG no tenía fondo transparente
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
    return img

def overlay_transparent(bg_img, img_overlay, x, y):
    """Superpone una imagen con transparencia sobre otra en las coordenadas (x,y)."""
    bg_h, bg_w, bg_channels = bg_img.shape
    ov_h, ov_w, ov_channels = img_overlay.shape

    # Calcular coordenadas seguras para evitar errores si la imagen sale del borde
    y1, y2 = max(0, y), min(bg_h, y + ov_h)
    x1, x2 = max(0, x), min(bg_w, x + ov_w)

    if y1 >= y2 or x1 >= x2:
        return bg_img # La imagen está completamente fuera de la pantalla

    y1_o, y2_o = max(0, -y), ov_h - max(0, (y + ov_h) - bg_h)
    x1_o, x2_o = max(0, -x), ov_w - max(0, (x + ov_w) - bg_w)

    # Separar el canal alfa (transparencia)
    alpha_s = img_overlay[y1_o:y2_o, x1_o:x2_o, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    # Mezclar los colores
    for c in range(0, 3):
        bg_img[y1:y2, x1:x2, c] = (alpha_s * img_overlay[y1_o:y2_o, x1_o:x2_o, c] +
                                   alpha_l * bg_img[y1:y2, x1:x2, c])
    return bg_img

def are_wrists_under_chin(pose_landmarks, face_landmarks):
    """Verifica si ambas muñecas están juntas y debajo de la barbilla."""
    # El nodo 152 corresponde al punto inferior de la barbilla en FaceMesh
    chin_node = face_landmarks.landmark[152] 
    
    left_wrist = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
    
    # Distancia entre ambas muñecas (para validar que estén "juntas")
    wrists_dist = math.sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)
    
    # Centro geométrico entre ambas muñecas
    mid_wrist_x = (left_wrist.x + right_wrist.x) / 2
    mid_wrist_y = (left_wrist.y + right_wrist.y) / 2
    
    # Distancia del centro de las muñecas a la barbilla
    dist_to_chin = math.sqrt((mid_wrist_x - chin_node.x)**2 + (mid_wrist_y - chin_node.y)**2)
    
    # En OpenCV/MediaPipe, Y crece hacia abajo. Por lo tanto, > significa "debajo de".
    # Agregamos los márgenes lógicos
    are_below = left_wrist.y > chin_node.y and right_wrist.y > chin_node.y
    are_together = wrists_dist < 0.15 # Ajusta este umbral si requieres que estén más o menos pegadas
    are_close_to_chin = dist_to_chin < 0.25 # Margen de proximidad a la cara
    
    return are_below and are_together and are_close_to_chin

# -----------------------------------------------------------------------------

def main():
    print("🎥 Iniciando cámara... Presiona 'q' para salir.")

    # --- CARGAR LA IMAGEN PNG EN MEMORIA (Reemplaza la ruta por la tuya) ---
    overlay_image = load_png("Reconocimiento_facial\imagenes_recursos\Ahgggg.png", target_width=150)
    if overlay_image is None:
        return # Salir si no hay imagen
    # -------------------------------------------------------------------------

    try:
        cap = init_camera()

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7 
        ) as holistic:

            frames_gesto_perdido = 0
            MAX_FRAMES_PERDIDOS = 10 
            gesto_activo = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Efecto espejo ANTES de procesar
                frame = cv2.flip(frame, 1)

                image, results = process_frame(frame, holistic)
                image = draw_landmarks(image, results)

                gesture_detected_now = False
                
                # Validación del nuevo gesto (muñecas bajo la barbilla)
                if results.face_landmarks and results.pose_landmarks:
                    gesture_detected_now = are_wrists_under_chin(results.pose_landmarks, results.face_landmarks)

                # Lógica del Amortiguador (Buffer)
                if gesture_detected_now:
                    gesto_activo = True
                    frames_gesto_perdido = 0 
                else:
                    if gesto_activo:
                        frames_gesto_perdido += 1
                        if frames_gesto_perdido > MAX_FRAMES_PERDIDOS:
                            gesto_activo = False

                # --- PROYECCIÓN DEL PNG SOBRE LA CABEZA ---
                if gesto_activo and results.face_landmarks:
                    # Punto 10 es el centro superior de la frente
                    top_head = results.face_landmarks.landmark[10]
                    
                    # Convertir coordenadas normalizadas a píxeles de pantalla
                    head_x = int(top_head.x * FRAME_WIDTH)
                    head_y = int(top_head.y * FRAME_HEIGHT)

                    ov_h, ov_w = overlay_image.shape[:2]

                    # Calcular posición (centrado en X, y encima de la cabeza en Y)
                    pos_x = head_x - (ov_w // 2)
                    pos_y = head_y - ov_h - 20 # 20 píxeles más arriba para no tapar la cara

                    # Superponer la imagen estática
                    image = overlay_transparent(image, overlay_image, pos_x, pos_y)
                # ------------------------------------------

                cv2.imshow("Detector de Gestos - MediaPipe", image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("🧹 Recursos liberados correctamente.")

if __name__ == "__main__":
    main()