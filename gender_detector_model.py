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
from joblib import load

detector = MTCNN()

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "gender_classification.z")

if not os.path.exists(model_path):
    print(f"[ERROR] Model file '{model_path}' not found. Please train the model with training.py first.")
    exit(1)

clf = load(model_path)

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
        cv2.rectangle(img, (x,y),(x+w, y+h),(0, 255, 0),2)

        # جلوگیری از منفی شدن مختصات
        x = max(0, x)
        y = max(0, y)

        return img[y:y+h, x:x+w], img, x , y, w, h
    except:
        pass

test_dir = os.path.join(base_dir, "test_model")
for i, item in enumerate(glob.glob(os.path.join(test_dir, "*"))):

    img = cv2.imread(item)
    

    face, x, y, w, h = face_detector(img)

    if face is None:
        continue
    
    face = cv2.resize(face, (32, 32))
    face = face.flatten()
    face = face/255
    
    out = clf.predict(np.array(face))[0]
    if out =="men":
      cv2.rectangle(img, (x,y),(x+w, y+h),(0, 0, 255),2)
      cv2.putText("men",(x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
    
    elif out =="women":
      cv2.rectangle(img, (x,y),(x+w, y+h),(0, 255, 0),2)
      cv2.putText("women",(x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

      
    cv2.imshow("image",img)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()    
    


  