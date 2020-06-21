import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from os import listdir
from datetime import datetime
from load_vgg import loadVggFaceModel
import pickle


#Paths Of Data
employee_pictures = "D:/sgp_ml/face_recognition/Data" #./Data"

detector = MTCNN()
color = (255, 0, 0)


# preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
# Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
    # Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
    img = preprocess_input(img)
    return img


print("loading model")
model = loadVggFaceModel()
print("loading done")
# # put your employee pictures in this path as name_of_employee.jpg

employees = dict()

# unknown = {}

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclidianDistance(source_representation, test_representation):
    dist = np.linalg.norm(source_representation - test_representation)
    return dist

for file in listdir(employee_pictures):
    employee, extension = file.split(".")
    employees[employee] = model.predict(preprocess_image(
        '%s/%s.%s' % (employee_pictures, employee, extension)))[0, :]
try:

    cap = cv2.VideoCapture(0)  # webcam
    if not cap:
        print("not cap")

    while(True):
        ret, img = cap.read()
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(frame)
        # print(faces)

        for face in faces:
            if face['confidence'] < .99:
                continue
            x, y, w, h = face['box']
            if w > 10:
                # draw rectangle to main image
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.imshow('img', img)

                detected_face = frame[int(y):int(y + h), int(x):int(x + w)]
                try:
                    detected_face = cv2.resize(
                        detected_face, (224, 224))  # resize to 224x224
                except:
                    continue

                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                # img_pixels /= 255
                # employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
                img_pixels /= 127.5
                img_pixels -= 1

                captured_representation = model.predict(img_pixels)[0, :]

                found = 0
                for i in employees:
                    employee_name = i
                    representation = employees[i]

                    print(i)
                    similarity = findCosineSimilarity(
                        representation, captured_representation)
                    print(similarity)
                    if(similarity < 0.30):
                        cv2.putText(img, employee_name, (int(
                            x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                        found = 1
                        cv2.imshow('img', img)
                    #else:
                    #    cv2.putText(img, 'Unknown', (int(
                     #       x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                        found = 0
                        cv2.imshow('img', img)

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

