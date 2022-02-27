# to detect only the lane lines we are using the bitwise and operation between the orinal image the masked images
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
class KalmanFilter:
    kf = cv2.KalmanFilter(4,2,0)
    kf.measurementMatrix = np.array([[1.3, 0, 0, 0], [0, 1.1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1.5], [0, 0, 1.07, 0], [0, 0, 0, .94]], np.float32)
    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y

kf = KalmanFilter()
def canny (image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur= cv2.GaussianBlur (gray, (5, 5),0)
    canny = cv2.Canny(blur, 50, 150, apertureSize=3)
    return canny

def make_coordinates (image, line_parameters):
    slope, intercept = line_parameters
    y1=image.shape [0]#left corner number of rows
    y2 = int(y1*(3/6))
    x1 = int((y1 - intercept)/slope)#  x=(y-c)/m
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


global_left_fit_average = []
global_right_fit_average = []
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    global global_left_fit_average
    global global_right_fit_average

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if (slope < 0):
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if (len(left_fit) == 0):
        left_fit_average = global_left_fit_average
    else:
        left_fit_average = np.average(left_fit, axis=0)
        global_left_fit_average = left_fit_average
    if (len(right_fit) == 0):
        right_fit_average = global_right_fit_average
    else:
        right_fit_average = np.average(right_fit, axis=0)
        global_right_fit_average = right_fit_average

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def display_lines (image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
       for line in lines:
           x1, y1, x2, y2 = line.reshape (4)  #reshape to convert 2D to 1D gives co-ordinates
           cv2.line(line_image,(x1,y1),(x2,y2), (0, 255, 0), 5)#draw the line b/w 2 points
    return line_image


def region_of_interest(image):
     #height = image.shape[0]
     polygons = np.array([[(60, 640), (635, 275), (1110, 670)]])
     mask = np.zeros_like(image)
     cv2.fillPoly(mask, polygons, 255)
     masked_image = cv2.bitwise_and(image, mask)
     return masked_image


cap=cv2.VideoCapture("6.mp4")
while(cap.isOpened()):
    _,frame=cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    # line_image = display_lines(frame,lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("Hough Transform Probablistic", combo_image)
    cv2.waitKey(1)
