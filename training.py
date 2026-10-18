import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import glob
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

detector = MTCNN()

data = []
labels = []

def face_detector(img):
    try: 
        
        # if img is None:   
        #     return None
        

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = detector.detect_faces(rgb_img)[0]

        # # اگر چهره‌ای پیدا نشد
        # if len(results) == 0:
        #     return None

        x, y, w, h = results["box"]

        # جلوگیری از منفی شدن مختصات
        x = max(0, x)
        y = max(0, y)

        return img[y:y+h, x:x+w]
    except:
        pass

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "gender_dataset")
files = glob.glob(os.path.join(dataset_dir, "*", "*"))

if len(files) == 0:
    print(f"[WARNING] No dataset found in '{dataset_dir}'. Please place gender_dataset folder here.")
    exit(0)

for i, item in enumerate(files):

    img = cv2.imread(item)

    face = face_detector(img)

    if face is None:
        continue
    
    face = cv2.resize(face, (32, 32))
    face = face.flatten()
    face = face/255
    
    data.append(face)
    label = item.split("\\")[-2]
    labels.append(label)
     
    if i % 100 == 0:
        print(f"[INFO]: {i}/3300 processed")
        
data = np.array(data)  
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

clf = SGDClassifier()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)


print(f"accuracy:{acc*100:.2f}")

dump(clf, "gender_classification.z")



    # cv2.imshow("image", face)
    # key = cv2.waitKey(0)
        
    # if key == 27 or key == ord('q'):
        
    #    break
  # don't run this code