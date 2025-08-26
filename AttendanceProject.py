import cv2
import numpy as np
import face_recognition
import os

# Find images and create a list for images and their names
path = 'ImagesAttendance/'
imagesList = []
classNames = []
myList = os.listdir(path)

# Read every el in the ImgAtt dir in order to organize them
for cl in myList:
    currentImage = cv2.imread(path + cl)
    imagesList.append(currentImage)
    classNames.append(os.path.splitext(cl)[0])

# func that will transform GBR to RGB, find faces' locations and their encodes
def findEncodings(images_list):
    encodeList = []
    for image in images_list:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodeList.append(face_recognition.face_encodings(image)[0])

    return encodeList

encodeListKnown = findEncodings(imagesList)
print("Encoding completed")

# ID = 0
cap = cv2.VideoCapture(0)

# Infinite while to get every frame
while(True):
    success, imageCam = cap.read()
    imgCamSmall = cv2.resize(imageCam, (0, 0), None, 0.25, 0.25) # 1/4 of the original image
    imgCamSmall = cv2.cvtColor(imgCamSmall, cv2.COLOR_BGR2RGB)

    facesInCurrFrame = face_recognition.face_locations(imgCamSmall) # All locations in the frame
    CurrFrameEncoding = face_recognition.face_encodings(imgCamSmall, facesInCurrFrame)

    for encodeFace, faceLoc in zip(CurrFrameEncoding, facesInCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        distances = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(distances)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(imageCam, (y1, x2), (y2, x1), (0, 255, 0), 2)
            cv2.rectangle(imageCam, (x1,y2-35), (x2, x2), (0, 255, 0), cv2.FILLED)
            cv2.putText(imageCam,name, (x1+6,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow('Video', imageCam)
    cv2.waitKey(1)