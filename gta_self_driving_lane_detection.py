import math
import cv2
import os
import time
import numpy as np
import pyautogui
import pandas as pd
import random
from PIL import ImageGrab


def input_to_key_strokes(prediction):
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

def lane_detection(lane_1, lane_2, final_score):
    predictions = [1, 0, 0, 0]

    if final_score < 75:
        predictions[0] = 0
        predictions[2] = 1
        predictions[1] = 1
    else:
        slope_lane_1 = ((lane_1[2] - lane_1[0]) / (lane_1[3] - lane_1[1]))
        slope_lane_2 = ((lane_2[2] - lane_2[0]) / (lane_2[3] - lane_2[1]))
        if abs(slope_lane_1 + 1.35) > 5:
            # pyautogui.press('d')
            predictions[1] = 1
        else:
            predictions[1] = 0
        if abs(slope_lane_2 - 3.55) > 4:
            # pyautogui.press('q')
            predictions[3] = 1
        else:
            predictions[3] = 0

    return predictions

    
def main():
    pyautogui.click(100, 100)
    winname = "screen"
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 800, 232)
    screen = np.ones(shape=(600, 799))
    oldscreen = np.zeros(shape=(600, 799))
    frame_count = 0
    max_frame_count = 3
    score = 0
    final_score = 100
    while True:
        pyautogui.keyUp('d')
        pyautogui.keyUp('q')
        screen = np.array(ImageGrab.grab(bbox=(1, 30, 800, 630)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        lane_1, lane_2 = find_lanes(screen)
        draw_lines(screen, [lane_1, lane_2])
        
        slope_lane_1 = ((lane_1[2] - lane_1[0]) / (lane_1[3] - lane_1[1]))
        slope_lane_2 = ((lane_2[2] - lane_2[0]) / (lane_2[3] - lane_2[1]))

        if abs(slope_lane_1 + 1.35) > 0.01:
            print("slope_lane_1: %.2f" % slope_lane_1)
            print("lane_1[0] lane_1[2]: %d %d" % (lane_1[0], lane_1[2]))
        if abs(slope_lane_2 - 3.55) > 0.01:
            print("slope_lane_2: %.2f" % slope_lane_2)
            print("lane_2[0] lane_2[2]: %d %d" % (lane_2[0], lane_2[2]))
        
        lane_predictions = lane_detection(lane_1, lane_2, final_score)
        print("lane_predictions:")
        print(lane_predictions)
        input_to_key_strokes(lane_predictions)
        
        frame_compare = np.absolute(np.array(screen) - np.array(oldscreen))
        
        if frame_count == max_frame_count:
            final_score = score / (max_frame_count - 1)
            final_score /= (600 * 799)
            final_score *= 100
            score = 0
            print(final_score)
            frame_count = 0
      
            
        if frame_count > 0:
            score += np.count_nonzero(frame_compare != 0)
            
        cv2.imshow(winname, screen)
        oldscreen = screen
        frame_count += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
