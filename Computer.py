import pyautogui
from time import sleep
from DatasetHandler import Dataset

class Bot:
    def __init__(self, modes):
        self.modes = modes
        self.dt = Dataset('dataset/landmarks.csv', modes)
        if self.modes['Info']:
            print('Auto Computer bot has been initialized')
    
    def __SwipeDesktop(self, direction):
        if direction in ['right', 'left']:
            if self.modes['Info']:
                print('Swiping Desktop to', direction)
            pyautogui.hotkey('win', 'ctrl', direction)
            sleep(0.5)
        else:
            if self.modes['Info']:
                print('invalid desktop direction')
    
    def Click(self, type):
        if type == 'Single':
            if self.modes['Info']:
                print('Single clicking on current location')
            pyautogui.click()
        elif type == 'Double':
            if self.modes['Info']:
                print('Double clicking on current location')
            pyautogui.doubleClick()
        sleep(0.5)
    
    def MoveMouse(self, position, shape):
        size = self.__GetScreenSize()
        x = position[0] / shape[0] * size['Width'];
        y = position[1] / shape[1] * size['Height'];
        pyautogui.moveTo(x, y)
    
    def __GetScreenSize(self):
        return { 
            "Width": pyautogui.size()[0],
            "Height": pyautogui.size()[1]
        }
    
    def Handle(self, action, row):
        if action == 'Right':
            if self.modes['Desktop']:
                self.__SwipeDesktop('right')
        elif action == 'Left':
            if self.modes['Desktop']:
                self.__SwipeDesktop('left')
        elif self.modes['Unknown']:
            row.append(action)
            self.dt.WriteRow(row)
        elif self.modes['Known']:
            row.append("Class")
            self.dt.WriteRow(row)
            
            