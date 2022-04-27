import cv2
import math
import json
import numpy as np
from datetime import datetime
from tabulate import tabulate

from Computer import Bot
from AudioManager import MasterVolume
from HandGesture import PostureDetector
from ImageProcessor import HandDetector

settings = json.load(open('Config/config.json'))

config = settings["Settings"]
gestureModels = settings["Models"]

print()
print('=====================')
print('|   Current Config   |')
print('=====================')
print('_____________________')
print(tabulate({"Mode": config.keys(), "Status": config.values()}, headers="keys", tablefmt="pretty"))
print(end='\n\n')

frames = 0
skipFrames = 1
audio = MasterVolume()
bot = Bot(config)
gesture = PostureDetector('dataset/landmarks.csv', config, gestureModels)
hands = HandDetector(config)

cap = cv2.VideoCapture(config["Camera"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

last_time = datetime.now()
while cap.isOpened():
    success, image = cap.read()
    if not success:
        if config["Warning"]:
            print("Ignoring empty camera frame.")
        continue
    else:
        image = cv2.flip(image, 1)
        frames += 1

    if frames % skipFrames == 0:
        image, unknown, mapping, mp_hands = hands.DetectHands(image)

        if config['Posture'] and len(mapping) > 0 and len(unknown) > 0:
            if len(mapping[0]) == 21:
                shape = gesture.Predict(unknown[0], mapping[0])
                    
                if config['Mouse'] and shape == 'One':
                    bot.MoveMouse(mapping[0][mp_hands.HandLandmark.INDEX_FINGER_TIP], image.shape)
            
                bot.Handle(shape, unknown[0])
                cv2.putText(image, 'POS: ' + shape, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    if config['FPS']:
        fps = np.around(frames / (datetime.now() - last_time).total_seconds(), 1)
        cv2.putText(image, 'FPS: ' + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    if config['LiveView']:
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
if config['LiveView']:
    cv2.destroyAllWindows()