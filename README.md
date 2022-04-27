# Hand Gesture Detector
###### NOTE: Project contains many bugs and not stable yet, needs a lot of improvements
## Description
The project is using [mediapipe](https://google.github.io/mediapipe/ "mediapipe") from google, The hands are detected using the mediapipe framework, after hands detection the gestures are detected to perform different operations available in the project.

Detection is performed using one of either two ways, that can be configured using the config file present in the code
- Using Machine Learning
- Using Classic Detection (Non ML)

If machine learning is being used to detect the gestures then the following classes are trained and available for detection.

> One, Two, Three, Four, Five

If Classic detector is being used to detect the hand posture or gesture then the following classes are available.

> One, Two, Three, Four, Five, Fist, Right, Left, Rocking, F\*\*k Of

## Requirements

  - Python >= 3.7
  - Mediapipe
  - Tensorflow

## Features in Development

Using the above gestures currently i am working on the following feature set to be used in the project
- Mouse Control using Hand Gesture (Index Finger)
- Audio Volume Control
- Windows Desktop Switch Left or Right
- Media Playback Track Change Next or Previous

## How to Run

To start the project, first clone the repo and then navigate to the project directory, then finally run the command

`pip3 install -r requirements.txt`

Once all the requirements are meet, then the simple command to run the project is following

`py start.py`

start.py is the entry file of the project it will automatically configure the parameters defined in the config file and show the preview window once initialized successfully!

## Config Params

Following parameters are configureable in the project via config file available in the config folder.

```json
{
     "Settings": {
          "Hands"    : true,
          "Detect"   : true,
          "FPS"      : true,
          "Mouse"    : false,
          "Media"    : false,
          "Desktop"  : false,
          "LiveView" : true,
          "Posture"  : true,
          "Unknown"  : false,
          "Known"    : false,
          "UseML"    : false,
          "Debug"    : false,
          "Info"     : true,
          "Warning"  : true,
          "Camera"   : 0
     },
     "Models": [
          {"Alias": "DT",    "Name": "Decision Tree",                   "Status": false },
          {"Alias": "KNN", "Name": "K-Nearest Neighbors",      "Status": false },
          {"Alias": "RF",     "Name": "Random Forest",               "Status": false },
          {"Alias": "SVM", "Name": "Support Vector Machine", "Status": true },
          {"Alias": "XGB",  "Name": "XG Boost",                         "Status": false },
          {"Alias": "NB",    "Name": "Naive Bayes",                    "Status": false}
     ]
}
```

## Explanation of Config Params

- Hands    (Show/Hide Hands and mapping points in live view)
- Detect   (Enable/Disable the hands and gesture detection)
- FPS      (Show/Hide the camera FPS value in the live view)
- Mouse    (Enable/Disable mouse control)
- Media    (Enable/Disable media playback control, e.g. play, pause next, previous)
- Desktop  (Enable/Disable multi desktop switch in windows)
- LiveView (Show/Hide Live view from the webcam and all processings)
- Posture  (Show/Hide the posture detected of the hand)
- Unknown  (Log the unknown entries in the dataset file)
- Known    (Log the known entries in the Dataset file)
- UseML    (Enable/Disable machine learning for posture detection)
- Debug    (Show/Hide debug messages in the console)
- Info     (Show/Hide info messages in the console)
- Warning  (Show/Hide warning messages in the console)
- Camera   (choose default camera device index, e.g. 0)

In models section simply you can enable or disable specific machine learning model if ML mode has been enabled in the config for posture detection.

