import math
import cv2
import os
import cvlib as cv
import argparse
from cvlib.object_detection import draw_bbox
import time
import numpy as np
import pyautogui
from sklearn import metrics, preprocessing
import pandas as pd
from PIL import ImageGrab
from pynput.keyboard import Key, Listener, KeyCode
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from scipy import stats
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean
import csv

def on_press(key):
    if key == KeyCode.from_char('z'):
        keyboard_input[0] = 1
    if key == KeyCode.from_char('d'):
        keyboard_input[1] = 1
    if key == KeyCode.from_char('s'):
        keyboard_input[2] = 1
    if key == KeyCode.from_char('q'):
        keyboard_input[3] = 1

def on_release(key):
    if key == KeyCode.from_char('z'):
        keyboard_input[0] = 0
    if key == KeyCode.from_char('d'):
        keyboard_input[1] = 0
    if key == KeyCode.from_char('s'):
        keyboard_input[2] = 0
    if key == KeyCode.from_char('q'):
        keyboard_input[3] = 0

def train_data(data, model_filename, weights_filename, write_to_file=False):
    data.loc[data.label != 'none', 'label'] = 1
    data.loc[data.label == 'none', 'label'] = 0
    X_column_names = data.columns.values[1:-4]
    Y_column_names = data.columns.values[-4:]
    X = data[X_column_names]
    Y = data[Y_column_names]
    print(X)
    print(Y)
    model = Sequential()
    model.add(Dense(12, input_dim=5, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=1000, batch_size=100)

    _, accuracy = model.evaluate(X, Y)
    print("model accuracy: %.2f" % accuracy)

    if write_to_file:
        model_json = model.to_json()
        with open(model_filename, 'w') as json_file:
            json_file.write(model_json)

        print("save model to disk")
        model.save_weights(weights_filename)
    return model

def train_lines_data(data, model_filename, weights_filename, write_to_file=False):
    X_column_names = data.columns.values[1:-4]
    Y_column_names = data.columns.values[-4:]
    print(X_column_names)
    print(Y_column_names)
    X = data[X_column_names]
    Y = data[Y_column_names]
    print(X)
    print(Y)
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=1000, batch_size=50)

    _, accuracy = model.evaluate(X, Y)
    print("model accuracy: %.2f" % accuracy)

    if write_to_file:
        model_json = model.to_json()
        with open(model_filename, 'w') as json_file:
            json_file.write(model_json)

        print("save model to disk")
        model.save_weights(weights_filename)
    return model


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,255,0], 3)
    except Exception as e:
        print("Exception occured in draw_lines: " + str(e))
        pass

def find_lanes(screen):
    try:
        processed_img = cv2.Canny(screen, threshold1=50, threshold2=300)
        processed_img = cv2.GaussianBlur(processed_img, (3,3), 0 )
        vertices = np.array([[10,500],[10,300], [300,250], [500,250], [800,300], [800,500]], np.int32)
        # vertices_line_1 = np.array([[0, 507], [213, 349], [234, 349], [14, 512]])
        processed_img  = roi(processed_img, [vertices])
        lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 20, 15)
        distances = []
        if lines is not None:
            for line in lines:
                coords = line[0]
                distances.append(dist(coords[0], coords[1], coords[2], coords[3]))


        # best_coords = [146, 490, 389, 310]
        best_coords = [389, 310, 146, 490]
        second_best_coords = [588, 303, 794, 361]
        if len(distances) >= 2:
            index_sorted_distances = np.argsort(distances)
            for index in index_sorted_distances:
                coords = lines[index][0]
                min_x = np.minimum(coords[0], coords[2])
                max_x = np.maximum(coords[0], coords[2])
                slope = 0
                if (coords[3] - coords[1]) != 0:
                    slope = ((coords[2] - coords[0]) / (coords[3] - coords[1]))

                if np.isfinite(slope):
                    if slope < 7 and slope > 3 and min_x > 400 and max_x > 600:
                        second_best_coords = coords
                    if slope > -7 and slope < -3 and max_x < 400 and min_x < 200:
                        best_coords = coords
        elif len(distances) == 1:
            index_sorted_distances = np.argsort(distances)
            for index in index_sorted_distances:
                coords = lines[index][0]
                min_x = np.minimum(coords[0], coords[2])
                max_x = np.maximum(coords[0], coords[2])
                slope = 0
                if (coords[3] - coords[1]) != 0:
                    slope = ((coords[2] - coords[0]) / (coords[3] - coords[1]))
                if np.isfinite(slope):
                    if slope > -7 and slope < -3 and max_x < 400 and min_x < 200:
                        best_coords = coords

        return best_coords, second_best_coords
    except Exception as e:
        print("Exception occured in find_lanes: " + str(e))
        pass

def predictions_to_key_strokes(predictions):
    for prediction in predictions:
        if prediction[0]:
            pyautogui.keyDown('z')
        else:
            pyautogui.keyUp('z')
        if prediction[1]:
            pyautogui.keyDown('d')
        else:
            pyautogui.keyUp('d')
        if prediction[2]:
            pyautogui.keyDown('s')
        else:
            pyautogui.keyUp('s')
        if prediction[3]:
            pyautogui.keyDown('q')
        else:
            pyautogui.keyUp('q')

def object_detection_training_data(screen, training_data):
    bbox, label, conf = cv.detect_common_objects(screen, enable_gpu=False, model='yolov3-tiny')
    output_image = draw_bbox(screen, bbox, label, conf)
    if len(label) > 0:
        for i in range(len(label)):
            training_data = training_data.append(dict(zip(training_data.columns.values, bbox[i] + [label[i]] + keyboard_input)), ignore_index=True)
    return training_data, output_image

def lane_detection_training_data(screen, training_data, lines_data_file_name):
    lane_1, lane_2 = find_lanes(screen)
    lines = np.array([lane_1, lane_2])
    draw_lines(screen, [lane_1, lane_2])
    training_data = training_data.append(dict(zip(training_data.columns.values, np.concatenate((np.reshape(np.array(lines), lines.shape[0] * lines.shape[1]), keyboard_input), axis=None))), ignore_index=True)
    return training_data

def object_detection(screen, training_data, model):
    start_time = time.time();
    bbox, label, conf = cv.detect_common_objects(screen, enable_gpu=False, model='yolov3-tiny')
    output_image = draw_bbox(screen, bbox, label, conf)
    end_time = time.time();
    print("detection time: " + str(end_time - start_time))
    prediction_data = pd.DataFrame()
    if (len(label)) > 0:
        for i in range(0, len(label)):
            # print(training_data.columns.values[0:-4])
            prediction_data = prediction_data.append(dict(zip(training_data.columns.values[0:-4], bbox[i] + [1])), ignore_index=True)
    else:
        prediction_data = prediction_data.append(dict(zip(training_data.columns.values[0:-4], [0, 0, 0, 0] + [0])), ignore_index=True)
    start_time = time.time()
    predictions = model.predict(prediction_data) > 0.5
    end_time = time.time()
    print("prediction time: " + str(end_time - start_time))
    print(predictions)
    return output_image, predictions

def lane_detection(screen, lines_model):
    columns = []
    for i  in range(0, 8):
        columns.append("x" + str(i))
    prediction_data = pd.DataFrame(columns=columns)
    lane_1, lane_2 = find_lanes(screen)
    draw_lines(screen, [lane_1, lane_2])
    lines = np.array([lane_1, lane_2])
    prediction_data = prediction_data.append(dict(zip(columns, np.reshape(lines, lines.shape[0] * lines.shape[1]))), ignore_index=True)
    start_time = time.time()
    predictions = lines_model.predict(prediction_data) > 0.5
    end_time = time.time()
    print("prediction time: " + str(end_time - start_time))
    print(predictions)
    return predictions

def main():
    parser = argparse.ArgumentParser(description='general script for generating, training and predicting self driving behavior for GTA V')

    parser.add_argument('-o', '--objectdetection', help='filename for object detection', required=False, default='')
    parser.add_argument('-l', '--lanedetection', help='filename for lane detection', required=False, default='')
    parser.add_argument('-t', '--task', help='set what task the script needs to perform: [generate, train, predict]', required=True)
    args = parser.parse_args()
    object_detection_file = args.objectdetection
    lane_detection_file = args.lanedetection
    task = args.task

    global keyboard_input
    keyboard_input = [0, 0, 0, 0]
    listener = Listener(on_press=on_press, on_release=on_release)

    obj_data_file_name = 'training_data/{}_train.csv'.format(object_detection_file)
    lines_data_file_name = 'training_data/{}_train.csv'.format(lane_detection_file)

    obj_predict_file_name = 'prediction_data/{}_predict.csv'.format(object_detection_file)
    lines_predict_file_name = 'prediction_data/{}_predict.csv'.format(lane_detection_file)

    obj_model_file_name = 'models/{}.json'.format(object_detection_file)
    lines_model_file_name = 'models/{}.json'.format(lane_detection_file)

    obj_weights_file_name = 'models/{}.h5'.format(object_detection_file)
    lines_weights_file_name = 'models/{}.h5'.format(lane_detection_file)

    obj_training_data = pd.DataFrame(columns=['x', 'y', 'w', 'h', 'label', 'z', 'd', 's', 'q'])
    columns = []
    for i  in range(0, 8):
        columns.append("x" + str(i))
    columns = columns + ['z', 'd', 's', 'q']
    lines_training_data = pd.DataFrame(columns=columns)

    obj_model = None
    lines_model = None

    if lane_detection_file != '':
        if task == 'predict':
            if os.path.isfile(lines_model_file_name):
                print("lines model file exists! loading model form file!")
            else:
                raise Exception("file {} does not exist, abort!".format(lines_model_file_name))
            json_file = open(lines_model_file_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            lines_model = model_from_json(loaded_model_json)
            lines_model.load_weights(lines_weights_file_name)
            lines_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            print("loaded lines model from disk")
        elif task == 'train':
            if os.path.isfile(lines_data_file_name):
                print('lines training data file exists, loading previous data!')
            else:
                raise Exception("file {} does not exist, abort!".format(lines_data_file_name))
            lines_training_data = pd.read_csv(lines_data_file_name, sep=",")
            print(lines_training_data)
            lines_model = train_lines_data(lines_training_data, lines_model_file_name, lines_weights_file_name, write_to_file=True)
            X_column_names = lines_training_data.columns.values[1:-4]
            Y_column_names = lines_training_data.columns.values[-4:]
            X = lines_training_data[X_column_names]
            Y = lines_training_data[Y_column_names]
            predictions = lines_model.predict(X) > 0.5
            predict_data = pd.DataFrame()
            for prediction in predictions:
                predict_data = predict_data.append(dict(zip(['z', 'd', 's', 'q'], prediction)), ignore_index=True)
            print(predictions)
            print(predict_data)
            predict_data.to_csv(lines_predict_file_name)
            _, accuracy = lines_model.evaluate(X, Y)
            print("accuracy: %.2f" % accuracy)
        else:
            print('lines training data file does not exist, starting fresh!')

    if object_detection_file != '':
        if task == 'predict':
            if os.path.isfile(obj_model_file_name):
                print("object model file exists! loading model form file!")
            else:
                raise Exception("file {} does not exist, abort!".format(obj_model_file_name))
            json_file = open(obj_model_file_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            obj_model = model_from_json(loaded_model_json)
            obj_model.load_weights(obj_weights_file_name)
            obj_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            print("loaded model from disk")
        elif task == 'train':
            if os.path.isfile(obj_data_file_name):
                print('object training data file exists, loading previous data!')
            else:
                raise Exception("file {} does not exist, abort!".format(obj_data_file_name))
            obj_training_data = pd.read_csv(obj_data_file_name, sep=",")
            obj_model = train_data(obj_training_data, obj_model_file_name, obj_weights_file_name, write_to_file=True)
            obj_training_data.loc[obj_training_data.label != 'none', 'label'] = 1
            obj_training_data.loc[obj_training_data.label == 'none', 'label'] = 0
            X_column_names = obj_training_data.columns.values[1:-4]
            Y_column_names = obj_training_data.columns.values[-4:]
            X = obj_training_data[X_column_names]
            Y = obj_training_data[Y_column_names]
            predictions = obj_model.predict(X) > 0.5
            predict_data = pd.DataFrame()
            for prediction in predictions:
                predict_data = predict_data.append(dict(zip(['z', 'd', 's', 'q'], prediction)), ignore_index=True)
            print(predictions)
            print(predict_data)
            predict_data.to_csv(obj_predict_file_name)
            _, accuracy = obj_model.evaluate(X, Y)
            print("accuracy: %.2f" % accuracy)
        else:
            print('object training data file does not exist, starting fresh!')

    if task != 'train':
        pyautogui.click(100, 100)
        winname = "screen"
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 800, 232)
        listener.start()
        while True:
            screen = np.array(ImageGrab.grab(bbox=(1, 30, 800, 630)))
            if task == 'predict':
                if object_detection_file != '':
                    object_detection_image, object_detection_predictions = object_detection(screen, obj_training_data, obj_model)
                    predictions_to_key_strokes(object_detection_predictions)

                if lane_detection_file != '':
                    lane_detection_predictions = lane_detection(screen, lines_model)
                    if lane_detection_predictions is not None:
                        predictions_to_key_strokes(lane_detection_predictions)

            if task == 'generate':
                if lane_detection_file != '':
                    lines_training_data = lane_detection_training_data(screen, lines_training_data, lines_data_file_name)
                    print(lines_training_data.shape)

                if object_detection_file != '':
                    obj_training_data, object_detection_training_image = object_detection_training_data(screen, obj_training_data)
                    print(obj_training_data.shape)

            cv2.imshow(winname, screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                if task == 'generate':
                    if object_detection_file != '':
                        obj_training_data.to_csv(obj_data_file_name)
                    if lane_detection_file != '':
                        lines_training_data.to_csv(lines_data_file_name)
                listener.stop()
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    main()
