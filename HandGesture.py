import csv
import numpy as np
from sklearn import metrics
import mediapipe as mp
import matplotlib.pyplot as plt
from DatasetHandler import Dataset

from xgboost import XGBClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

class PostureDetector:
    def __init__(self, filename, modes, detectors, transform=False):
        self.filename = filename
        self.transform = transform
        self.modes = modes
        
        self.models = {
            "DT": DecisionTreeClassifier(max_depth = 100),
            "KNN": KNeighborsClassifier(n_neighbors = 15),
            "RF": RandomForestClassifier(n_estimators=150),
            "SVM": CalibratedClassifierCV(SVC(kernel = 'rbf', C = 10)),
            "XGB": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            "NB": GaussianNB()
        }
        
        for detector in detectors:
            if detector["Status"] == False:
                if modes["Info"]:
                    print(detector["Name"], "has been disabled")
   
                del self.models[detector["Alias"]]
            else:
                if modes["Info"]:
                    print(detector["Name"], "is being used")
        
        self.scale = False
        if transform:
            self.scale = True
            if self.modes['Info']:
                print('Data transformation has been enabled')

        self.InitTraining()
        if self.models is not None:
            for model in self.models.keys():
                if modes['Debug']:
                    print(model, 'Parameters currently in use:\n')
                    print(self.models[model].get_params())
            
            if modes['Info']:    
                print('Hand Gesture detector is ready using', filename)
        else:
            if modes['Warning']:
                print('Hand Gesture is not ready!')
        self.predictor = ClassicShapePredictor(modes)
    
    def Predict(self, unknown, mapping):
        if self.modes["UseML"]:
            if self.models is not None:
                unknown = np.array(unknown).reshape((1,-1))
                if self.transform:
                    unknown = self.scaler.fit_transform(unknown)
            
                res = []
                for model in self.models.keys():
                    result = list(self.models[model].predict_proba(unknown)[0])
                    res.append(self.models[model].classes_[result.index(max(result))] if max(result) >= 0.9 else -1)
            
                if self.modes['Debug']:
                    print(res)
            
                majority = Counter(res).most_common()[0][0]
                return self.classes[majority] if majority != -1 else 'Unknown'
            else:
                return 'Unknown'
        else:
            return self.predictor.Predict(mapping)

    def InitTraining(self, show=False):
        db = Dataset(self.filename, self.modes)
        tupple = db.ReadCSV(self.scale)
        if tupple:
            X_train, X_test, y_train, y_test = train_test_split(tupple[0], tupple[1], random_state=42, test_size=0.3)
            for model in self.models.keys():
                self.models[model].fit(X_train, y_train)
                if self.modes['Debug']:
                    y_pred=self.models[model].predict(X_test)
                    print("Accuracy", model, ": %.2f" % metrics.accuracy_score(y_test, y_pred))
                
                if show:
                    plot_confusion_matrix(self.models[model], X_test, y_test)  
                    plt.show()
        else:
            self.models = None

class ClassicShapePredictor:
    def __init__(self, modes):
        self.modes = modes
        self.mp_hands = mp.solutions.hands
        self.shapes = ['One', 'Two', 'Three', 'Four', 'Five', 'Fist', 'Left', 'Right', 'Fuck Of', 'Rocking']
        if modes['Info']:
            print('Classic Shape predictor is ready')
            print('Available Classes', self.shapes)
    
    def __Distance(self, point1, point2):
        if point1 is not None and point2 is not None:
            return int(np.linalg.norm(np.array(point1) - np.array(point2)))
        else:
            if self.modes["Warning"]:
                print('Invalid landmark points for distance measuring')
            return -1;
    
    def __IsFist(self, mapping):
        points = []
        points.append(self.__Distance(mapping['index'], mapping['middle']))
        points.append(self.__Distance(mapping['ring'], mapping['pinky']))
        isUp = self.__IsUP(mapping, ['index', 'middle', 'ring', 'pinky'])
        distThumb = self.__Distance(mapping['index'], mapping['thumb'])
        return abs(max(points) - min(points)) <= 3 and isUp == False and distThumb >= 100
    
    def __IsUP(self, mapping, names):
        criteria = []
        for key in mapping.keys():
            if key != 'thumb':
                if key in names:
                    criteria.append(mapping[key][1] < mapping['thumb'][1])
                else:
                    criteria.append(mapping[key][1] > mapping['thumb'][1])
                
        return all(criteria)
    
    def __IsRight(self, mapping):
        points = []
        points.append(self.__Distance(mapping['index'], mapping['middle']))
        points.append(self.__Distance(mapping['ring'], mapping['pinky']))
        isUp = self.__IsUP(mapping, ['index', 'middle', 'ring', 'pinky'])
        distThumb = self.__Distance(mapping['index'], mapping['thumb'])
        direction = mapping['index'][0] < mapping['thumb'][0]
        return abs(max(points) - min(points)) <= 3 and isUp == False and distThumb < 100 and direction
    
    def __IsLeft(self, mapping):
        points = []
        points.append(self.__Distance(mapping['index'], mapping['middle']))
        points.append(self.__Distance(mapping['ring'], mapping['pinky']))
        isUp = self.__IsUP(mapping, ['index', 'middle', 'ring', 'pinky'])
        distThumb = self.__Distance(mapping['index'], mapping['thumb'])
        direction = mapping['index'][0] > mapping['thumb'][0]
        return abs(max(points) - min(points)) <= 3 and isUp == False and distThumb < 100 and direction
    
    def Predict(self, mapping):
        try:
            mappings = {
                "thumb": mapping[self.mp_hands.HandLandmark.THUMB_TIP],
                "index": mapping[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
                "middle": mapping[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                "ring":  mapping[self.mp_hands.HandLandmark.RING_FINGER_TIP],
                "pinky": mapping[self.mp_hands.HandLandmark.PINKY_TIP]
            }
        
            if self.__IsFist(mappings):
                return self.shapes[5]
            elif self.__IsUP(mappings, ['index']):
                return self.shapes[0]
            elif self.__IsUP(mappings, ['index', 'middle']):
                return self.shapes[1]
            elif self.__IsUP(mappings, ['index', 'middle', 'ring']):
                return self.shapes[2]
            elif self.__IsUP(mappings, ['index', 'middle', 'ring', 'pinky']) and abs(mappings['pinky'][0] - mappings['thumb'][0]) <= 5:
                return self.shapes[3]
            elif self.__IsUP(mappings, ['index', 'middle', 'ring', 'pinky']) and abs(mappings['pinky'][0] - mappings['thumb'][0]) >= 15:
                return self.shapes[4]
            elif self.__IsLeft(mappings):
                return self.shapes[6]
            elif self.__IsRight(mappings):
                return self.shapes[7]
            elif self.__IsUP(mappings, ['middle']):
                return self.shapes[8]
            elif self.__IsUP(mappings, ['index', 'pinky']):
                return self.shapes[9]
            else:
                return 'Unknown'
        except:
            return ''
        