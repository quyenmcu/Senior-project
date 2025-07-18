import os
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
from collections import deque
from tensorflow.lite.python.interpreter import Interpreter
import csv

# Setup directories
def setup_user_directories(user_id):
    user_dir = os.path.join("data", user_id)
    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(os.path.join(user_dir, "captured_images"), exist_ok=True)
    os.makedirs(os.path.join(user_dir, "recorded_videos"), exist_ok=True)
    return user_dir

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh

# Load TensorFlow Lite model
interpreter = Interpreter(model_path="mobilenetv2_finetuned.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess frame for the model
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=0).astype(np.float32)

# Function to compute Euclidean distance
def compute(ptA, ptB):
    return np.linalg.norm(np.array(ptA) - np.array(ptB))

# Compute Eye Aspect Ratio (EAR) using eye landmarks
def compute_ear(eye_landmarks):
    vertical1 = compute(eye_landmarks[1], eye_landmarks[5])
    vertical2 = compute(eye_landmarks[2], eye_landmarks[4])
    horizontal = compute(eye_landmarks[0], eye_landmarks[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# Yawning detection using mouth landmarks
def is_yawning(mouth_landmarks):
    if len(mouth_landmarks) >= 4:
        top_lip = compute(mouth_landmarks[0], mouth_landmarks[1])
        bottom_lip = compute(mouth_landmarks[2], mouth_landmarks[3])
        ratio = top_lip / bottom_lip
        return ratio > 0.6
    return False

# Function to save an annotated picture
def save_annotated_picture(frame, user_dir, status):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(user_dir, "captured_images", f"{status}_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Picture saved: {filename}")

# Initialize video writer
def init_video_writer(frame, user_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(user_dir, "recorded_videos", f"fatigue_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    height, width = frame.shape[:2]
    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
    print(f"Recording started: {filename}")
    return out

# CSV logging function
def log_to_csv(user_dir, data):
    csv_path = os.path.join(user_dir, "drowsiness_log.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Status"])
        writer.writerow(data)

# Main detection function
def drowsiness_detection(user_id, cap):
    user_dir = setup_user_directories(user_id)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
    status, last_status = "", None
    color = (0, 0, 0)
    
    # Smoothing windows for EAR and mouth aspect ratio (MAR)
    ear_history = deque(maxlen=20)
    mar_history = deque(maxlen=20)
    recording = False
    video_writer = None

    # Blink parameters
    blink_threshold = 0.22         # EAR value below which the eye is considered “closed”
    blink_duration_frames = 3      # Number of consecutive frames below threshold needed to consider it prolonged (i.e. not just a blink)
    blink_counter = 0              # Count of consecutive frames below threshold

    # Initialize last_valid_ear with a default value (will be updated during calibration)
    last_valid_ear = None

    # Calibration phase: compute the average EAR over several frames
    calibration_frames = 50
    avg_ear, frame_count = 0.0, 0
    print("Calibrating... Please look directly at the camera.")
    while frame_count < calibration_frames:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            left_eye_pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h))
                            for i in [33, 160, 158, 133, 153, 144]]
            right_eye_pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h))
                             for i in [362, 385, 387, 263, 373, 380]]
            left_ear = compute_ear(left_eye_pts)
            right_ear = compute_ear(right_eye_pts)
            avg_ear += (left_ear + right_ear) / 2.0
    avg_ear /= calibration_frames
    print(f"Calibration complete. Average EAR: {avg_ear:.3f}")
    last_valid_ear = avg_ear  # initialize with the calibrated value

    # Main detection loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Default state is Active
        status = "Active"
        color = (0, 255, 0)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            left_eye_pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h))
                            for i in [33, 160, 158, 133, 153, 144]]
            right_eye_pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h))
                             for i in [362, 385, 387, 263, 373, 380]]
            mouth_pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h))
                         for i in [13, 14, 78, 308]]

            # Compute EAR for each eye and then the average
            left_ear = compute_ear(left_eye_pts)
            right_ear = compute_ear(right_eye_pts)
            current_ear = (left_ear + right_ear) / 2.0

            # --- Modified logic to reduce blink impact ---
            # If the current EAR is below the blink threshold:
            if current_ear < blink_threshold:
                blink_counter += 1
                # If the low EAR is only for a brief moment (a blink),
                # use the last valid EAR instead of the very low value.
                if blink_counter < blink_duration_frames:
                    ear_history.append(last_valid_ear)
                else:
                    # If the low EAR persists, update with the current value.
                    ear_history.append(current_ear)
                    last_valid_ear = current_ear
            else:
                # Eye is open: update normally and reset the blink counter.
                blink_counter = 0
                last_valid_ear = current_ear
                ear_history.append(current_ear)
            # Compute the smoothed EAR from the history.
            smoothed_ear = np.mean(ear_history)
            # --- End modified logic ---

            # Compute mouth aspect ratio (MAR)
            mar = compute(mouth_pts[0], mouth_pts[1]) / compute(mouth_pts[2], mouth_pts[3])
            mar_history.append(mar)
            smoothed_mar = np.mean(mar_history)

            # State determination using smoothed EAR and MAR.
            # (Threshold multipliers are relative to the calibrated avg_ear.)
            if smoothed_ear < avg_ear * 0.65:
                status = "Sleeping"
                color = (0, 0, 255)
            elif avg_ear * 0.65 <= smoothed_ear < avg_ear * 0.85:
                status = "Drowsy"
                color = (0, 255, 255)
            elif smoothed_mar > 0.6:
                status = "Yawning"
                color = (0, 165, 255)

            # Draw a bounding box around the face.
            x_min = int(min([lm.x for lm in face_landmarks]) * w)
            y_min = int(min([lm.y for lm in face_landmarks]) * h)
            x_max = int(max([lm.x for lm in face_landmarks]) * w)
            y_max = int(max([lm.y for lm in face_landmarks]) * h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            # Draw landmarks for eyes and mouth.
            for point in left_eye_pts + right_eye_pts:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)
            for point in mouth_pts:
                cv2.circle(frame, point, 2, (255, 0, 0), -1)

            # Display debugging information.
            cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Save annotated image when drowsiness or sleep is detected.
            if status in ["Drowsy", "Sleeping"] and not recording:
                save_annotated_picture(frame, user_dir, status)

            # Start video recording if fatigue is detected.
            if status in ["Drowsy", "Sleeping"] and not recording:
                recording = True
                video_writer = init_video_writer(frame, user_dir)

            # Stop recording when back to Active.
            if recording and status == "Active":
                recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                print("Recording stopped.")

            if recording and video_writer:
                video_writer.write(frame)

            # Log any state change to CSV.
            if status != last_status:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_to_csv(user_dir, [timestamp, status])
                last_status = status

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    if video_writer:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    user_id = input("Enter User ID: ")
    cap = cv2.VideoCapture(0)
    drowsiness_detection(user_id, cap)
