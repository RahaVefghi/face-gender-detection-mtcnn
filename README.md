# Face & Gender Detection System with MTCNN & Machine Learning

> 🎓 **Academic Note:** This project was developed as a practical assignment for the **Machine Learning & Deep Learning Course** to explore deep face detection models (MTCNN), facial region extraction, and lightweight classification algorithms.

---

## 📌 Project Overview

Facial attribute analysis is a cornerstone of modern computer vision applications. This project implements a modular, two-stage vision pipeline that detects faces in unconstrained images and subsequently classifies each detected face by gender (`men` vs. `women`):

1. **Robust Face Detection via MTCNN:**
   - Deep **Multi-task Cascaded Convolutional Networks (MTCNN)** localize faces across varied poses, lighting conditions, and scales.
   - Automatic bounding box coordinate validation prevents negative coordinate slicing errors.
2. **Preprocessing & Gender Classification:**
   - Face crops are standardized to $32 \times 32$ pixels and normalized to the $[0, 1]$ floating-point range.
   - An optimized **SGDClassifier (Stochastic Gradient Descent with linear SVM/log loss)** provides lightweight, lightning-fast inference suitable for embedded edge devices.
   - Model persistence using `joblib` serialization (`gender_classification.z`).
3. **Real-Time Visual Overlay:**
   - OpenCV overlays bounding boxes and gender labels onto image frames with proper color indication.

---

## 🏗️ Pipeline Flow

```mermaid
flowchart LR
    A["Input Image"] --> B["MTCNN Face Detector"]
    B --> C["Cropped Face Region"]
    C --> D["Resize (32x32) & Normalization"]
    D --> E["SGD Classifier"]
    E --> F["Visual Output with Bounding Box"]
```

---

## 📂 Project Structure

```text
face-gender-detection-mtcnn/
│
├── gender_detector_model.py   # Inference script: Detects faces and predicts gender on test images
├── training.py                # Pipeline script: Detects faces from dataset and trains the classifier
├── test_model/                # Diverse test images for immediate evaluation
├── requirements.txt           # Project dependencies
├── .gitignore                 # Files excluded from version control
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/RahaVefghi/face-gender-detection-mtcnn.git
cd face-gender-detection-mtcnn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run inference on test images
Run the model across test images in `test_model/`:
```bash
python gender_detector_model.py
```

### 4. Retrain the model on your dataset
To train the classifier on your dataset (structured with `Men` and `Women` subfolders):
```bash
python training.py
```

---

## 🛠️ Technologies Used
- **Python 3.10+**
- **MTCNN (Multi-task Cascaded CNN):** Deep facial detection and landmark extraction
- **OpenCV (cv2):** Image manipulation, scaling, and visualization overlays
- **Scikit-Learn:** SGDClassifier, metric evaluation, train/test split
- **Joblib:** Lightweight model persistence
