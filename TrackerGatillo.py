import cv2
import mediapipe as mp
import math
import numpy as np
from PIL import Image, ImageSequence

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

# -------------------  FUNCIONES DE VALIDACION  ----------------------
def load_gif_frames(gif_path, target_width=150):
    """Carga un GIF, redimensiona los frames y los convierte a formato OpenCV."""
    try:
        gif = Image.open(gif_path)
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo: {gif_path}")
        return []

    frames = []
    for frame in ImageSequence.Iterator(gif):
        # Asegurar que tenga fondo transparente
        frame = frame.convert("RGBA")
        
        # Calcular nueva altura manteniendo la proporción
        aspect_ratio = frame.height / frame.width
        target_height = int(target_width * aspect_ratio)
        frame = frame.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Convertir a arreglo de numpy para OpenCV (de RGBA a BGRA)
        cv_frame = np.array(frame)
        cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_RGBA2BGRA)
        frames.append(cv_frame)
    return frames

def overlay_transparent(bg_img, img_overlay, x, y):
    """Superpone una imagen con transparencia sobre otra en las coordenadas (x,y)."""
    bg_h, bg_w, bg_channels = bg_img.shape
    ov_h, ov_w, ov_channels = img_overlay.shape

    # Calcular coordenadas seguras para evitar errores si el GIF sale del borde
    y1, y2 = max(0, y), min(bg_h, y + ov_h)
    x1, x2 = max(0, x), min(bg_w, x + ov_w)

    if y1 >= y2 or x1 >= x2:
        return bg_img # El GIF está completamente fuera de la pantalla

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

def is_wrist_at_eyebrows(wrist_landmark, face_landmarks):
    # Puntos clave del rostro para delimitar "la zona de la frente"
    eyebrow_y = face_landmarks.landmark[9].y      
    face_left_x = face_landmarks.landmark[234].x  
    face_right_x = face_landmarks.landmark[454].x 

    # Coordenadas de la muñeca (viene del modelo de Pose, no de Hand)
    wrist_y = wrist_landmark.y
    wrist_x = wrist_landmark.x

    # Aumentamos un poco el margen vertical porque la muñeca está más abajo que los dedos
    is_at_height = abs(wrist_y - eyebrow_y) < 0.20

    # Margen horizontal
    min_x = min(face_left_x, face_right_x) - 0.1 
    max_x = max(face_left_x, face_right_x) + 0.1
    is_in_front_of_face = min_x < wrist_x < max_x

    return is_at_height and is_in_front_of_face

def is_pinch(hand_landmarks):
    # Obtener las coordenadas del pulgar (4) y el índice (8)
    thumb_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]

    # Calcular la distancia euclidiana entre ambos puntos
    distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

    # Si la distancia es muy pequeña, los dedos se están tocando
    return distance < 0.05

def is_covering_mouth(hand_landmarks, face_landmarks):
    # Punto 13 es el centro del labio superior. Punto 9 de la mano es el centro de la palma.
    mouth_y = face_landmarks.landmark[13].y
    mouth_x = face_landmarks.landmark[13].x
    palm_y = hand_landmarks.landmark[9].y
    palm_x = hand_landmarks.landmark[9].x

    # Calculamos la distancia entre la palma y la boca
    distance = math.sqrt((mouth_x - palm_x)**2 + (mouth_y - palm_y)**2)
    
    # Si la distancia es menor a 0.1, consideramos que la mano está sobre la boca
    return distance < 0.1

def is_hand_at_eyebrows(hand_landmarks, face_landmarks):
    # Puntos clave del rostro para delimitar "la zona de la frente"
    eyebrow_y = face_landmarks.landmark[9].y      # Altura entre las cejas
    face_left_x = face_landmarks.landmark[234].x  # Lado izquierdo del rostro
    face_right_x = face_landmarks.landmark[454].x # Lado derecho del rostro

    # Centro de la mano
    palm_y = hand_landmarks.landmark[9].y
    palm_x = hand_landmarks.landmark[9].x

    # Condición A: La mano está a la altura de las cejas (+/- un pequeño margen vertical)
    is_at_height = abs(palm_y - eyebrow_y) < 0.15

    # Condición B: La mano está horizontalmente dentro del ancho de la cara
    # Usamos min() y max() porque dependiendo de la cámara, la izquierda y derecha se invierten
    min_x = min(face_left_x, face_right_x) - 0.05
    max_x = max(face_left_x, face_right_x) + 0.05
    is_in_front_of_face = min_x < palm_x < max_x

    return is_at_height and is_in_front_of_face

def main():
    print("🎥 Iniciando cámara... Presiona 'q' para salir.")

    # --- CARGAR EL GIF EN MEMORIA (Reemplaza 'animacion.gif' con tu archivo) ---
    gif_frames = load_gif_frames("Reconocimiento_facial\imagenes_recursos\Gatillo.gif", target_width=150)
    if not gif_frames:
        return # Salir si no hay GIF
    
    current_frame_idx = 0
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

                # ¡CAMBIO CRUCIAL! Hacemos el efecto espejo ANTES de procesar
                frame = cv2.flip(frame, 1)

                image, results = process_frame(frame, holistic)
                image = draw_landmarks(image, results)

                gesture_detected_now = False
                
                # Validación del gesto
                if results.face_landmarks and results.pose_landmarks:
                    face = results.face_landmarks
                    pose = results.pose_landmarks
                    
                    left_wrist = pose.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
                    right_wrist = pose.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]

                    case_1 = False
                    case_2 = False

                    if results.left_hand_landmarks:
                        case_1 = is_covering_mouth(results.left_hand_landmarks, face) and is_wrist_at_eyebrows(right_wrist, face)

                    if results.right_hand_landmarks:
                        case_2 = is_covering_mouth(results.right_hand_landmarks, face) and is_wrist_at_eyebrows(left_wrist, face)

                    if case_1 or case_2:
                        gesture_detected_now = True

                # Lógica del Amortiguador
                if gesture_detected_now:
                    gesto_activo = True
                    frames_gesto_perdido = 0 
                else:
                    if gesto_activo:
                        frames_gesto_perdido += 1
                        if frames_gesto_perdido > MAX_FRAMES_PERDIDOS:
                            gesto_activo = False

                # --- PROYECCIÓN DEL GIF SOBRE LA CABEZA ---
                if gesto_activo and results.face_landmarks:
                    # Punto 10 es el centro superior de la frente
                    top_head = results.face_landmarks.landmark[10]
                    
                    # Convertir coordenadas normalizadas a píxeles de pantalla
                    head_x = int(top_head.x * FRAME_WIDTH)
                    head_y = int(top_head.y * FRAME_HEIGHT)

                    # Obtener el fotograma actual del GIF
                    overlay_img = gif_frames[current_frame_idx]
                    ov_h, ov_w = overlay_img.shape[:2]

                    # Calcular posición (centrado en X, y encima de la cabeza en Y)
                    pos_x = head_x - (ov_w // 2)
                    pos_y = head_y - ov_h - 20 # 20 píxeles más arriba para no taparte la cara

                    # Superponer la imagen
                    image = overlay_transparent(image, overlay_img, pos_x, pos_y)

                    # Avanzar al siguiente frame del GIF para crear la animación
                    current_frame_idx = (current_frame_idx + 1) % len(gif_frames)
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