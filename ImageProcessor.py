import cv2
import numpy as np
import mediapipe as mp

class HandDetector:
    def __init__(self, modes, complexity=0, threshold=0.5):
        self.modes = modes
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=complexity, min_detection_confidence=threshold, min_tracking_confidence=threshold)
        if self.modes["Info"]:
            print('Hand Detector is ready')

    def __Distance(self, point1, point2):
        if point1 is not None and point2 is not None:
            return int(np.linalg.norm(np.array(point1) - np.array(point2)))
        else:
            if self.modes["Warning"]:
                print('Invalid landmark points for distance measuring')
            return -1;
    
    def DetectHands(self, image):
        unknowns = []
        mappings = []
        if self.modes['Detect']:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    unknown = []
                    mapping = {}
                    for point in self.mp_hands.HandLandmark:
                        normalizedLandmark = hand_landmarks.landmark[point]
                        pixelCoordinatesLandmark = self.mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, image.shape[0], image.shape[1])
                        mapping[point] = pixelCoordinatesLandmark
                
                    if mapping[self.mp_hands.HandLandmark.WRIST] is not None:  
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.WRIST], mapping[self.mp_hands.HandLandmark.THUMB_IP]))
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.WRIST], mapping[self.mp_hands.HandLandmark.THUMB_TIP]))
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.WRIST], mapping[self.mp_hands.HandLandmark.INDEX_FINGER_DIP]))
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.WRIST], mapping[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]))
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.WRIST], mapping[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP]))
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.WRIST], mapping[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]))
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.WRIST], mapping[self.mp_hands.HandLandmark.RING_FINGER_DIP]))
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.WRIST], mapping[self.mp_hands.HandLandmark.RING_FINGER_TIP]))
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.WRIST], mapping[self.mp_hands.HandLandmark.PINKY_DIP]))
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.WRIST], mapping[self.mp_hands.HandLandmark.PINKY_TIP]))
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.THUMB_TIP], mapping[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]))
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.THUMB_TIP], mapping[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]))
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.THUMB_TIP], mapping[self.mp_hands.HandLandmark.RING_FINGER_TIP]))
                        unknown.append(self.__Distance(mapping[self.mp_hands.HandLandmark.THUMB_TIP], mapping[self.mp_hands.HandLandmark.PINKY_TIP]))
                    
                    if self.modes['Hands']:
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    mappings.append(mapping)
                    if len(unknown) > 0:
                        unknowns.append(unknown)
                        
        return (image, unknowns, mappings, self.mp_hands)