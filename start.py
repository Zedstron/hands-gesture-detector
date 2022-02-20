import cv2
import math
import numpy as np
from datetime import datetime
from tabulate import tabulate

from Computer import Bot
from AudioManager import MasterVolume
from HandGesture import PostureDetector
from ImageProcessor import HandDetector

modes = {
    "Hands"    : True,
    "Detect"   : True,
    "FPS"      : True,
    "Mouse"    : False,
    "Media"    : False,
    "Desktop"  : False,
    "LiveView" : True,
    "Posture"  : True,
    "Unknown"  : False,
    "Known"    : False,
    "UseML"    : False,
    "Debug"    : False,
    "Info"     : True,
    "Warning"  : True
}

print()
print('=====================')
print('|   Current Modes   |')
print('=====================')
print('_____________________')
print(tabulate({"Mode": modes.keys(), "Status": modes.values()}, headers="keys", tablefmt="pretty"))
print(end='\n\n')

gestureModels = [
    {"Alias": "DT",  "Name": "Decision Tree",          "Status": True },
    {"Alias": "KNN", "Name": "K-Nearest Neighbors",    "Status": True },
    {"Alias": "RF",  "Name": "Random Forest",          "Status": True },
    {"Alias": "SVM", "Name": "Support Vector Machine", "Status": True },
    {"Alias": "XGB", "Name": "XG Boost",               "Status": True },
    {"Alias": "NB",  "Name": "Naive Bayes",            "Status": False}
]

frames = 0
skipFrames = 1
audio = MasterVolume()
bot = Bot(modes)
gesture = PostureDetector('landmarks.csv', modes, gestureModels)
hands = HandDetector(modes)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

last_time = datetime.now()
while cap.isOpened():
    success, image = cap.read()
    if not success:
        if modes["Warning"]:
            print("Ignoring empty camera frame.")
        continue
    else:
        image = cv2.flip(image, 1)
        frames += 1

    if frames % skipFrames == 0:
        image, unknown, mapping, mp_hands = hands.DetectHands(image)

        if modes['Posture'] and len(mapping) > 0 and len(unknown) > 0:
            if len(mapping[0]) == 21:
                shape = gesture.Predict(unknown[0], mapping[0])
                    
                if modes['Mouse'] and shape == 'One':
                    bot.MoveMouse(mapping[0][mp_hands.HandLandmark.INDEX_FINGER_TIP], image.shape)
            
                bot.Handle(shape, unknown[0])
                cv2.putText(image, 'POS: ' + shape, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    if modes['FPS']:
        fps = np.around(frames / (datetime.now() - last_time).total_seconds(), 1)
        cv2.putText(image, 'FPS: ' + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    if modes['LiveView']:
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
if modes['LiveView']:
    cv2.destroyAllWindows()