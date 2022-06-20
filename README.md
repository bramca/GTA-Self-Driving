# GTA V Self Driving
This repo contains `python` scripts that try learn the machine to drive a car in GTA V.

# Models
There are 3 models in this repo:
- [gta_self_driving.py](./gta_self_driving.py)
  This is a machine learning model where you first generate training and test data via playing the game.
  Then this model will train on this data and try to teach itself to drive.
  To use this model enter command `python gta_self_driving.py [-o|--objectdetectin] <object detection filename> [-l|--lanedetection] <lane detection filename> [-t|--task] <generate, train, predict>`
  requirements:
  ```
  numpy
  pyautogui
  sklearn
  pandas
  keras
  scipy
  numpy
  cvlib
  Pillow
  pyinput
  ```

- [gta_self_driving_evolution_algo.py](./gta_self_driving_evolution_algo.py)
  This is a neuro evolution model that will try to start driving with a starting population of 20 drivers.
  It will give a score based on comparison between current and old frames, the more the frames are different the higher the score.
  e.g. the more it is driving the better score.
  Then it will create a new population based on the 2 best scoring drivers with some random mutations.
  To use this model enter command `python gta_self_driving_evolution_algo.py`
  requirements:
  ```
  cv2
  numpy
  pyautogui
  pandas
  Pillow
  ```

- [gta_self_driving_lane_detection.py](./gta_self_driving_lane_detection.py)
  This model is an attempt to create a lane detection algorithm using image processing.
  Then it tries to make self driving prediction analyzing the detected lanes.
  To use this model enter command `python gta_self_driving_lane_detection.py`
  requirements:
  ```
  cv2
  numpy
  pyautogui
  pandas
  Pillow
  ```
