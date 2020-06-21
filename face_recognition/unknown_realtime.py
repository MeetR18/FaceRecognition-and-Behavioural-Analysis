import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from os import listdir
from datetime import datetime
from load_vgg import loadVggFaceModel
import pymysql
import pickle


#Paths Of Data
unknown_path = "/mnt/592D2D01503E82C6/Projects/Ai/FaceDRR/Unknown"#./Unkown/"
employee_pictures = "/mnt/592D2D01503E82C6/Projects/Ai/FaceDRR/Data" #./Data"

#Database Info

db_host = "localhost"
db_username = "ml"
db_pass = "ml"
db_dbName = "old_face_recognition"
db_port = 3306


db = pymysql.connect(db_host, db_username, db_pass, db_dbName, port=db_port)

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
# put your employee pictures in this path as name_of_employee.jpg

employees = dict()

unknown = {}

for file in listdir(employee_pictures):
    employee, extension = file.split(".")
    employees[employee] = model.predict(preprocess_image(
        '%s/%s.png' % (employee_pictures, employee)))[0, :]

with open("/mnt/592D2D01503E82C6/Projects/Ai/FaceDRR/known_representation.txt", "wb") as myFile:
    pickle.dump(employees, myFile)


print("employee representations retrieved successfully")


def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclidianDistance(source_representation, test_representation):
    dist = np.linalg.norm(source_representation - test_representation)
    return dist

try:
    db = pymysql.connect(db_host,
                         db_username, db_pass, db_dbName, port=3306)
    cursor = db.cursor()
    Face_captured = {}
    Unknown_face_captured = {}

    cap = cv2.VideoCapture(0)  # webcam
    if not cap:
        print("not cap")

    while(True):
        ret, img = cap.read()
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(frame)

        for face in faces:
            if face['confidence'] < .99:
                continue
            x, y, w, h = face['box']
            if w > 10:
                # draw rectangle to main image
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow('img', img)

                # crop detected face
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

                now = datetime.now()
                j = now.strftime('%Y-%m-%d %H:%M:%S')
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

                        if employee_name in Face_captured.keys():
                            duration = now - Face_captured[employee_name][1]
                            duration = duration.seconds
                            Face_captured[employee_name][1] = now

                            if duration >= 60:
                                Face_captured[employee_name][0] = j

                                sql = "INSERT INTO `mmh2`(`name`, `firstTime`, `lastTime`) VALUES ('%s','%s','%s')" % (
                                    i, j, j)

                                try:
                                    # Execute the SQL command
                                    cursor.execute(sql)
                                    # Commit your changes in the database
                                    db.commit()
                                    # db.rollback()
                                except Exception as e:
                                    # Rollback in case there is any error
                                    db.rollback()
                            else:

                                sql = "UPDATE `mmh2` SET `lastTime`='%s' WHERE `name`='%s' and `firstTime`='%s'" % (
                                    j, i, Face_captured[employee_name][0])

                                try:
                                    # Execute the SQL command
                                    cursor.execute(sql)
                                    # Commit your changes in the database
                                    db.commit()
                                    # db.rollback()
                                except Exception as e:
                                    # Rollback in case there is any error
                                    db.rollback()

                        else:
                            Face_captured[employee_name] = [j, now]

                            sql = "INSERT INTO `mmh2`(`name`, `firstTime`, `lastTime`) VALUES ('%s','%s','%s')" % (
                                i, j, j)

                            try:
                                # Execute the SQL command
                                cursor.execute(sql)
                                # Commit your changes in the database
                                db.commit()
                            except Exception as e:
                                # Rollback in case there is any error
                                db.rollback()
                        break

                if(found == 0):  # if found image is not in Unkown database
                    unknown_found = 0
                    #unknown[id] = [captured_representation, j, now]

                    for i in unknown:
                        unknown_name = "unknown" + i
                        representation = unknown[i]

                        similarity = findCosineSimilarity(
                            representation, captured_representation)
                        print(similarity)
                        if(similarity < 0.30):
                            cv2.putText(img, unknown_name, (int(x), int(y)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            unknown_found = 1
                            cv2.imshow('img', img)

                            if unknown_name in Unknown_face_captured.keys():
                                duration = now - Unknown_face_captured[unknown_name][1]
                                duration = duration.seconds
                                Unknown_face_captured[unknown_name][1] = now

                                if duration >= 60:
                                    Unknown_face_captured[unknown_name][0] = j

                                    sql = "INSERT INTO `mmh2`(`name`, `firstTime`, `lastTime`) VALUES ('%s','%s','%s')" % (
                                        unknown_name, j, j)

                                    try:
                                        # Execute the SQL command
                                        cursor.execute(sql)
                                        # Commit your changes in the database
                                        db.commit()
                                        # db.rollback()
                                    except Exception as e:
                                        # Rollback in case there is any error
                                        db.rollback()
                                else:

                                    sql = "UPDATE `mmh2` SET `lastTime`='%s' WHERE `name`='%s' and `firstTime`='%s'" % (
                                        j, unknown_name, Unknown_face_captured[unknown_name][0])

                                    try:
                                        # Execute the SQL command
                                        cursor.execute(sql)
                                        # Commit your changes in the database
                                        db.commit()
                                        # db.rollback()
                                    except Exception as e:
                                        # Rollback in case there is any error
                                        db.rollback()

                            else:
                                Unknown_face_captured[unknown_name] = [j, now]

                                sql = "INSERT INTO `mmh2`(`name`, `firstTime`, `lastTime`) VALUES ('%s','%s','%s')" % (
                                    unknown_name, j, j)

                                try:
                                    # Execute the SQL command
                                    cursor.execute(sql)
                                    # Commit your changes in the database
                                    db.commit()



                                except Exception as e:
                                    # Rollback in case there is any error
                                    db.rollback()
                            break
                    if unknown_found == 0:
                        #newentry in unknownlist
                        sql = "SELECT MAX(id) FROM `unknown`"
                        mx_id = 0
                        try:
                            cursor.execute(sql)
                            mx_id = cursor.fetchall()[0][0]
                            db.commit()
                        except Exception as e:
                            db.rollback()

                        id = str(len(unknown) + 1)
                        new_id = "unknown" + id
                        unknown[id] = captured_representation
                        cv2.imwrite(unknown_path + new_id + j + ".jpg", detected_face)
                        cv2.putText(img, new_id, (int(x), int(y)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        Unknown_face_captured[new_id] = [j, now]

                        sql = "INSERT INTO `mmh2`(`name`, `firstTime`, `lastTime`) VALUES ('%s','%s','%s')" % (
                            new_id, j, j)

                        try:
                            # Execute the SQL command
                            cursor.execute(sql)
                            # Commit your changes in the database
                            db.commit()
                        except Exception as e:
                            # Rollback in case there is any error
                            db.rollback()

        cv2.imshow('img', img)
        # capture unknown

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

# kill open cv things
finally:
    cap.release()
    cv2.destroyAllWindows()
    # disconnect from server
db.close()
