Siddu P, [14-12-2021 10:36]
import cv2
import numpy as np

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
           cv2.line(line_image,(x1,y1),(x2,y2), (0, 0, 255), 5)#draw the line b/w 2 points
    return line_image

def kal_lines(image,lines):
    kal_image= np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)  # reshape to convert 2D to 1D gives co-ordinates
            predicted = kf.predict(x1, y1)
            predicted1=kf.predict(x2,y2)
            cv2.circle(kal_image, (x1, y1), 5, (0, 255, 0), 4)
            cv2.circle(kal_image, (x2, y2), 5, (0, 255, 0), 4)
            cv2.line(kal_image, (predicted[0] + 80, predicted[1]+50), (predicted1[0]+120,predicted1[1]+130), (0,255,0), 5)
            cv2.circle(kal_image, (predicted[0] + 80, predicted[1]+50), 5, (0, 0, 255), 4)
            cv2.circle(kal_image, (predicted1[0]+120,predicted1[1]+130), 5, (0, 0, 255), 4)
            # cv2.line(kal_image,(predicted[0]+50, predicted[1]),(x2, y2-10), (250, 100, 75), 5)
            # cv2.circle(kal_image, (predicted[0]+150, predicted[1]), 5, (255, 0,0), 4)
            #
            #     # predicted1 = kf.predict(x1, y1)
            #     # cv2.circle(kal_image, (predicted1[0], predicted1[1]), 5, (0, 0, 255), 4)
            #     # print(predicted1)
            # cv2.line(kal_image, (predicted[0], predicted[1]), (x2, y2), (255, 0, 0), 5)
            # cv2.line(kal_image, (predicted[0],predicted[1]), (predicted1[0], predicted1[1]), (255, 0, 0), 5)
    #
    return kal_image

Siddu P, [14-12-2021 10:36]
def region_of_interest(image):
     #height = image.shape[0]
     polygons = np.array([[(60, 640), (635, 275), (1110, 670)]])
     mask = np.zeros_like(image)
     cv2.fillPoly(mask, polygons, 255)
     masked_image = cv2.bitwise_and(image, mask)
     return masked_image


cap=cv2.VideoCapture("WhatsApp Video 2021-09-10 at 14.49.41.mp4")
while(cap.isOpened()):
    _,frame=cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    kal_image=kal_lines(region_of_interest(frame),averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    result= cv2.addWeighted(combo_image, 0.8, kal_image, 1, 1)
    cv2.imshow("result", result)
    cv2.waitKey(40)
