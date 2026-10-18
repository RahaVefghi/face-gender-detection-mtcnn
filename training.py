import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import glob
import cv2
import numpy as np
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

def extract_face(detector, img):
    if img is None:
        return None
    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_img)
        if len(results) == 0:
            return None

        x, y, w, h = results[0]["box"]
        x, y = max(0, x), max(0, y)
        face_crop = img[y:y + h, x:x + w]
        if face_crop.size == 0:
            return None
        return face_crop
    except Exception:
        return None

def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "gender_dataset")
    files = glob.glob(os.path.join(dataset_dir, "*", "*"))

    if len(files) == 0:
        print(f"[WARNING] No dataset found in '{dataset_dir}'.")
        print("Please place the 'gender_dataset' folder containing class subfolders to train.")
        return None, None, None, None

    detector = MTCNN()
    data = []
    labels = []
    total = len(files)
    print(f"[INFO] Detecting faces and loading {total} images...")

    for i, file_path in enumerate(files):
        img = cv2.imread(file_path)
        face = extract_face(detector, img)
        if face is None:
            continue

        face_resized = cv2.resize(face, (32, 32))
        normalized_features = (face_resized / 255.0).flatten()

        data.append(normalized_features)
        label = os.path.basename(os.path.dirname(file_path))
        labels.append(label)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"[INFO] {i + 1}/{total} images processed")

    data_arr = np.array(data)
    labels_arr = np.array(labels)
    return train_test_split(data_arr, labels_arr, test_size=0.2, random_state=42)

def train_and_save_model(x_train, x_test, y_train, y_test):
    print(f"[INFO] Training SGDClassifier on {len(x_train)} samples...")
    clf = SGDClassifier(loss="log_loss", random_state=42, max_iter=1000)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[EVALUATION] Validation Accuracy: {acc * 100:.2f}%")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(base_dir, "gender_classification.z")
    dump(clf, model_save_path)
    print(f"[INFO] Model successfully saved to {model_save_path}")

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_dataset()
    if x_train is not None and len(x_train) > 0:
        train_and_save_model(x_train, x_test, y_train, y_test)