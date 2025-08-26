import cv2
import numpy as np
import face_recognition
from face_recognition import face_distance

# Import and convert imgs from bgr to rgb
imgElon = face_recognition.load_image_file("ImagesBasic/elon_musk_image.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file("ImagesBasic/elon_musk_test.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Find the location of the face in the image
face_location = face_recognition.face_locations(imgElon)[0]
face_location_test = face_recognition.face_locations(imgTest)[0]

# Encode the face
encodeElon = face_recognition.face_encodings(imgElon)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]

# Results and distance (lower the distance better the match)
results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDistance = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDistance)
color = (0,255,0) if results[0] else (0,0,255)
cv2.putText(imgTest,f'{results} {round(faceDistance[0], 2)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Draw rectangles around the face
cv2.rectangle(imgElon, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (255, 0, 255), 2)
cv2.rectangle(imgTest, (face_location_test[3], face_location_test[0]), (face_location_test[1], face_location_test[2]), (255, 0, 255), 2)

cv2.imshow("Elon Musk", imgElon)
cv2.imshow("Test", imgTest)
cv2.waitKey(0)