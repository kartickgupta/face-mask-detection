import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

new_model = load_model("Mask Detection/Saved Model/face_mask-2.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

color_result_dict = {1 : ['With Mask', (0, 225,0 )],
                     0 : ['Without Mask', (0, 0, 225)]}

while(True):
    ret, frame = cam.read()
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    
    for(x, y, w, h) in faces:
        face_crop = frame[x: x+w, y: y+h]
        resized_face = cv2.resize(face_crop, (224, 224))
        normalized_face = resized_face / 225.0
        reshaped_face = normalized_face.reshape((1, 224, 224, 3))
        result = int(np.argmax(new_model.predict(reshaped_face), axis = 1))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color_result_dict[result][1], 2)
        cv2.putText(frame, text=color_result_dict[result][0], org=(x, y-40), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color= color_result_dict[result][1], thickness=2)


    cv2.imshow('Result image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()