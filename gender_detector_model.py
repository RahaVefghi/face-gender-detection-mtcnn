import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import glob
import cv2
import numpy as np
from mtcnn import MTCNN
from joblib import load

def detect_face(detector, img):
    if img is None:
        return None, None
    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_img)
        if len(results) == 0:
            return None, None

        x, y, w, h = results[0]["box"]
        x, y = max(0, x), max(0, y)
        face_crop = img[y:y + h, x:x + w]
        if face_crop.size == 0:
            return None, None
        return face_crop, (x, y, w, h)
    except Exception:
        return None, None

def run_inference():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "gender_classification.z")
    test_dir = os.path.join(base_dir, "test_model")

    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model file '{model_path}' not found.")
        print("Please train the model first by running training.py.")
        return

    test_files = glob.glob(os.path.join(test_dir, "*"))
    if len(test_files) == 0:
        print(f"[ERROR] No test images found in '{test_dir}'.")
        return

    print(f"[INFO] Loading MTCNN face detector and trained model from {model_path}...")
    detector = MTCNN()
    clf = load(model_path)
    print(f"[INFO] Evaluating {len(test_files)} test images. Press any key to see next, or 'q' to quit.\n")

    for file_path in test_files:
        img = cv2.imread(file_path)
        if img is None:
            continue

        face_crop, bbox = detect_face(detector, img)
        if face_crop is None:
            print(f"Sample: {os.path.basename(file_path):<30} -> No face detected.")
            continue

        x, y, w, h = bbox
        resized = cv2.resize(face_crop, (32, 32))
        features = (resized / 255.0).flatten().reshape(1, -1)

        predicted_gender = clf.predict(features)[0]
        color = (255, 105, 180) if "women" in str(predicted_gender).lower() else (0, 255, 0)

        # Draw bounding box and label
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(img, (x, max(0, y - 30)), (x + w, y), color, -1)
        cv2.putText(
            img, str(predicted_gender).capitalize(),
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        print(f"Sample: {os.path.basename(file_path):<30} -> Predicted: {predicted_gender}")
        cv2.imshow("Face & Gender Detection", img)
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()