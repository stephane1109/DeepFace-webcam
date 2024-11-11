# pip install deepface opencv-python-headless numpy pandas matplotlib requests tf_keras

import cv2
from deepface import DeepFace


# Fonction principale pour la détection d'émotions en utilisant DeepFace
def detect_emotions_deepface():
    cap = cv2.VideoCapture(0)

    # Réduire la résolution pour améliorer la fluidité
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # Forcer la fréquence d'images à 25 FPS
    cap.set(cv2.CAP_PROP_FPS, 25)

    # Obtenir et afficher la fréquence d'images de la webcam
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Fréquence d'images de la webcam : {fps} FPS")

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire une image depuis la webcam.")
            break

        # Effacer tous les anciens cadres verts en réinitialisant le cadre
        display_frame = frame.copy()

        # Conversion de l'image de BGR à RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Analyse des émotions avec MTCNN pour une meilleure précision
            resultats = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False,
                                         detector_backend='retinaface')

            # Afficher le dernier résultat d'émotion si disponible
            if isinstance(resultats, list):  # Plusieurs visages détectés
                for res in resultats:
                    x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                    emotion_dominante = res['dominant_emotion']
                    probabilite = res['emotion'][emotion_dominante]

                    # Dessiner un rectangle autour du visage et afficher l'émotion dominante
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{emotion_dominante}: {probabilite:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            else:  # Un seul visage détecté
                x, y, w, h = resultats['region']['x'], resultats['region']['y'], resultats['region']['w'], \
                resultats['region']['h']
                emotion_dominante = resultats['dominant_emotion']
                probabilite = resultats['emotion'][emotion_dominante]

                # Dessiner un rectangle autour du visage et afficher l'émotion dominante
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{emotion_dominante}: {probabilite:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print("Erreur d'analyse de DeepFace : ", str(e))

            # Afficher le flux vidéo avec les annotations
        cv2.imshow('Détection des émotions en temps réel', display_frame)

        # Appuyer sur 'q' ou 'ESC' pour quitter la boucle
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:  # 27 est le code pour la touche ESC
            break

        # Libérer les ressources et fermer les fenêtres
    cap.release()
    cv2.destroyAllWindows()


# Lancer la détection d'émotions
detect_emotions_deepface()
