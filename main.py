import cv2
import face_recognition
from deepface import DeepFace
import numpy as np

# === 1. Load and encode the known face ===
# Note the forward‑slash path so Python doesn’t interpret '\t' as a tab
known_image = face_recognition.load_image_file("images/target.png")
known_face_encoding = face_recognition.face_encodings(known_image)[0]
known_face_name = "Authorized User"

# === 2. Initialize webcam ===
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Optionally resize for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # === 3. Detect and recognize faces ===
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # scale back up to original frame size
        top, right, bottom, left = top*2, right*2, bottom*2, left*2
        face_img = frame[top:bottom, left:right]

        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        name = "Unknown"
        color = (0, 0, 255)

        if True in matches:
            # === 4. Liveness detection on the cropped face region ===
            try:
                faces = DeepFace.extract_faces(
                    img_path=face_img,
                    detector_backend='opencv',
                    align=False,
                    enforce_detection=False,
                    anti_spoofing=True
                )
                if faces and faces[0].get("is_real", False):
                    name = f"{known_face_name} (Live)"
                    color = (0, 255, 0)
                else:
                    name = "Spoof Detected"
            except Exception as e:
                print("Liveness check error:", e)
                name = "Liveness Error"

        # === 5. Draw box and label ===
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # === 6. Display ===
    cv2.imshow("Live Face Recognition + Liveness", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
