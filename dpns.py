import numpy as np
import cv2
import mediapipe as mp


def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(np.array([eye_points[1].x, eye_points[1].y]) - np.array([eye_points[5].x, eye_points[5].y]))
    B = np.linalg.norm(np.array([eye_points[2].x, eye_points[2].y]) - np.array([eye_points[4].x, eye_points[4].y]))
    C = np.linalg.norm(np.array([eye_points[0].x, eye_points[0].y]) - np.array([eye_points[3].x, eye_points[3].y]))
    return (A + B) / (2.0 * C)



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)


cap = cv2.VideoCapture(0)


EAR_THRESHOLD = 0.27
FRAME_THRESHOLD = 8
closed_frames = 0


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
NOSE_TIP = [1]
NOSE_LEFT = [98]
NOSE_RIGHT = [327]
MOUTH = [78, 81, 13, 311, 308, 402]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE]
            right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE]


            avg_EAR = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2


            if avg_EAR < EAR_THRESHOLD:
                closed_frames += 1
                if closed_frames >= FRAME_THRESHOLD:
                    cv2.putText(frame, "DROWSY! WAKE UP!", (170, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                closed_frames = 0


            for i in LEFT_EYE + RIGHT_EYE:
                point = face_landmarks.landmark[i]
                x, y = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


            cv2.putText(frame, f"EAR: {avg_EAR:.2f} (Threshold: {EAR_THRESHOLD})", (100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 1)


            nose_tip_point = face_landmarks.landmark[NOSE_TIP[0]]
            nose_tip_x, nose_tip_y = int(nose_tip_point.x * frame.shape[1]), int(nose_tip_point.y * frame.shape[0])
            cv2.circle(frame, (nose_tip_x, nose_tip_y), 2, (255, 0, 0), -1)


            nose_left_point = face_landmarks.landmark[NOSE_LEFT[0]]
            nose_left_x, nose_left_y = int(nose_left_point.x * frame.shape[1]), int(nose_left_point.y * frame.shape[0])
            cv2.circle(frame, (nose_left_x, nose_left_y), 2, (255, 0, 0), -1)


            nose_right_point = face_landmarks.landmark[NOSE_RIGHT[0]]
            nose_right_x, nose_right_y = int(nose_right_point.x * frame.shape[1]), int(nose_right_point.y * frame.shape[0])
            cv2.circle(frame, (nose_right_x, nose_right_y), 2, (255, 0, 0), -1)


            for i in MOUTH:
                point = face_landmarks.landmark[i]
                x, y = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)


    cv2.imshow('Drowsiness Detection with Nose and Mouth', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()