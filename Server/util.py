import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}
__model = None
def getBase64TestImgMsd():
    with open("base64rohit.txt") as f:
        return f.read()

def get_name_from_class_number(class_num):
    return __class_number_to_name[class_num]

def classifyImg(base64EncodedString, filePath=None):
    imgs = getCroppedImgIf2Eyes(filePath, base64EncodedString)
    resluts = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, "db1", 5)
        scalled_har_img = cv2.resize(img_har, (32, 32))

        # Reshape scalled_har_img to have the same number of dimensions as scalled_raw_img
        scalled_har_img = scalled_har_img.reshape(32 * 32, 1)

        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_har_img))

        len_img_array = 32 * 32 * 3 + 32 * 32
        final = combined_img.reshape(1, len_img_array).astype(float)
        resluts.append({
            "class": get_name_from_class_number(__model.predict(final)[0]),
            "class_proba": np.round(__model.predict_proba(final)*100,2).tolist()[0],
            "class_dict": __class_name_to_number
        })
    return resluts


def load_artifacts():
    print("loading saved artifacts... start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open("./artifacts/face_detection_model.pkl", "rb") as f:
            __model = joblib.load(f)
    print("loading saved artifacts, done")
def getCroppedImgIf2Eyes(imgPath, base64EncodedString):
    face_cascade = cv2.CascadeClassifier("../Model/opencv/haarcascades/haarcascade_frontalface_default.xml")
    eyes_cascade = cv2.CascadeClassifier("../Model/opencv/haarcascades/haarcascade_eye.xml")

    if imgPath:
        img = cv2.imread(imgPath)
    else:
        img = getCv2ImgFromBase64String(base64EncodedString)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    croppedFaces = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:
            croppedFaces.append(roi_color)
    return croppedFaces


def getCv2ImgFromBase64String(encodedString):
    encodedData = encodedString.split(",")[1]
    npArray = np.frombuffer(base64.b64decode(encodedData), np.uint8)
    img = cv2.imdecode(npArray, cv2.IMREAD_COLOR)
    return img

if __name__ == "__main__":
    load_artifacts()
    print(classifyImg(None, "./test_images/316605.jpg"))
