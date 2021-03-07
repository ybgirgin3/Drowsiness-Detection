import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# ses oynatma için 
# mixer.init()
# sound = mixer.Sound("alarm.wav")


# face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
face = cv2.CascadeClassifier(cv2.data.haarcascades + 'files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(cv2.data.haarcascades + 'files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(cv2.data.haarcascades + 'files/haarcascade_righteye_2splits.xml')

# label = ["open", "close"]
label = None

# model = load_model('trash/DD/models/cnncat2.h5')
model = load_model('cnncat2.h5')

# read camera
cap = cv2.VideoCapture(0)

# string variables
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# integer variables
count = 0
score = 0
thickness = 2


l_pred = [99]
r_pred = [99]


# main loop
while True:
    ret, frame = cap.read()
    height, weight = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(leye)
    right_eye =  reye.detectMultiScale(reye)

    # warning rectangle
    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED)

    # face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )



        # left eye
        for (x, y, w, h) in left_eye:
            # l > left eye
            l = frame[y:y+h, x:x+h]
            count += 1

            l = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
            l = cv2.resize(l, (24,24))
            l = l / 255
            l = l.reshape(24, 24, -1)
            l = np.expand_dims(l, axis=0)

            # prediction
            l_pred = model.predict_classes(l)
            # if l_pred[0] == 1 : return open
            # if l_pred[0] == 0 : return close
            # break


        # right eye
        for (x, y, w, h) in right_eye:
            # r > right eye

            r = frame[y:y+h, x:x+h]
            count += 1

            r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
            r = cv2.resize(r, (24, 24))
            r = r / 255
            r = r.reshape(24, 24, -1) 
            r = np.expand_dims(r, axis=0)

            # prediction
            r_pred = model.predict_classes(r)
            # if r_pred[0] == 1 : return open
            # if r_pred[0] == 0 : return close
            # break


        # prediction statements
        
        if r_pred[0] == 0 and l_pred[0] == 0:
            score += 1
            thickness += 2
            label = "Closed"

            if score < 0:
                score = 0

            elif score > 15:
                # took a pic of driver and beep the alarm
                cv2.imwrite(os.path.join(os.getcwd, "image.png"), frame)

                # if sound is playable
                try:
                    sound.play()
                except:
                    pass


        else: 
            score -= 1
            thickness -= 2
            label = "Open"

        cv2.putText(frame, label, (10, height-20), font, 1, (255, 255 ,255), 1, cv2.LINE_AA)

        cv2.putText(frame, "Score:"+str(score), (10, height-40), font, 1, (255, 255 ,255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thickness)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



       

















