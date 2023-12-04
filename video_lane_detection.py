import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def region_of_interest(image, vertices) :
    mask = np.zeros_like(image)
    match_mask_color = 255 
    cv.fillPoly(mask, vertices, match_mask_color)
    masked = cv.bitwise_and(image, mask)
    return masked

def get_x_y(image, lane) :

    height = image.shape[0]  
    y1 = height
    y2 = int(height / 2) 
    x1 = int((y1 - lane[0][1]) / lane[0][0])
    x2 = int((y2 - lane[0][1]) / lane[0][0])
    return [x1, y1, x2, y2]

def slope_line(image, lines):
    
    main_lane, right_lane, left_lane = [], [], []

    right_m, right_c, left_m, left_c = 0, 0, 0, 0

    mid = int(image.shape[1] / 2) 

    if lines is None :
        return []

    for line in lines:
        x1, y1,x2, y2 = line[0]
        if x1 == x2 :
            continue

        m, c = np.polyfit((x1, x2), (y1, y2), 1)
        if m > 0 and x1 > mid and x2 > mid :
            right_lane.append((m, c))

        elif m <= 0 and x1 < mid and x2 < mid  :
            left_lane.append((m, c))
    
    for right in right_lane:
        right_m += (right[0] / len(right_lane))
        right_c += (right[1] / len(right_lane))

    if (len(right_lane)):
        right_lane_final = [[right_m, right_c]] 

    for left in left_lane :
        left_m += (left[0] / len(left_lane))
        left_c += (left[1] / len(left_lane))

    if (len(left_lane)):
        left_lane_final = [[left_m, left_c]]
    
    if left_m != 0 :
        main_lane.append(get_x_y(image, left_lane_final))
        
    if right_m != 0 :
        main_lane.append(get_x_y(image, right_lane_final))
    
    if len(main_lane):
        return main_lane

    else :
        return []

    
def track_line(image, track_point):

    global new_steer_angle, steer_history

    steer_py, steer_px = image.shape
    blank_image = np.zeros(((image.shape[0]), image.shape[1], 3), dtype = np.uint8)
    steerx = 0
    
    if len(track_point) == 2 :
        for track in track_point:
            x1 = track[0] 
            x2 = track[2]
            y1 = track[1]
            y2 = track[3]
            steerx += track[2]
            cv.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 9)

        cv.line(blank_image, (int((steer_px / 2)), steer_py), (int(steerx / len(track_point)), int(steer_py / 2)), (0, 0, 255), 9)
        trace = cv.addWeighted(img, 0.8, blank_image, 1.2, 0.0)
        new_steer_angle = int((steerx / len(track_point)) * (180 / 640))
        steer_history.append(new_steer_angle)
        return trace

    elif len(track_point) == 1 :
        for track in track_point:
            x1 = track[0] 
            x2 = track[2]
            y1 = track[1]
            y2 = track[3]
            steerx += track[2]
            cv.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 9)
            m, c = np.polyfit((x1, x2), (y1, y2), 1)

        steerx = int((steer_py/2 - steer_py + m * (steer_px / 2)) / m)
        cv.line(blank_image, (int((steer_px / 2)), steer_py), (int(steerx / len(track_point)), int(steer_py / 2)), (0, 0, 255), 9)
        trace = cv.addWeighted(img, 0.8, blank_image, 1.2, 0.0)
        new_steer_angle = int((steerx / len(track_point)) * (180 / 640))
        steer_history.append(new_steer_angle)
        # print(pwm)
        return trace

    else :
        print(len(track_point))
        return img 

def steer_angle_stabilization(steer_history) :
    if len(steer_history) >= 2:
        print("yes")
        new_steer_angle = steer_history[-1]
        current_steer_angle = steer_history[-2]
        deviation = new_steer_angle - current_steer_angle
        if deviation > 10 or deviation < -10 :
            if deviation > 0 :
                steer_angle = current_steer_angle + 5

            elif deviation < 0 :
                steer_angle = current_steer_angle - 5

        else :
            steer_angle = new_steer_angle

    else :
        print("no")
        steer_angle = steer_history[-1]

    return steer_angle
 
region_of_interest_vertices = [
    (0, 480),
    (0, 232),
    (635, 234),
    (640, 480)
] 

steer_history = []
vi = cv.VideoCapture("track_test_01.mp4")
while vi.isOpened :

    flag, img = vi.read()
    img = cv.resize(img, (640, 480))

    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2] 

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lp = np.array([65, 40, 40])
    up = np.array([166, 255, 255])
    mask = cv.inRange(hsv, lp, up)
    res = cv.bitwise_and(img,img,mask=mask)
    cv.imshow('r ', res)
    edge = cv.Canny(res, 100, 210)
    cv.imshow('e ', edge)
    cropped_img = region_of_interest(edge, np.array([region_of_interest_vertices], np.int32))

    lines = cv.HoughLinesP(cropped_img, 2, np.pi / 180, 100, minLineLength= 30, maxLineGap= 10)

    track_point = slope_line(cropped_img, lines)

    track_image = track_line(cropped_img, track_point)

    steer_angle = steer_angle_stabilization(steer_history)
    print(steer_angle)
 
    cv.imshow("track", track_image)

    if cv.waitKey(15) == ord("q"):
        break

vi.release()
cv.destroyAllWindows()

