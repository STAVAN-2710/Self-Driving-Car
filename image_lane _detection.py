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

    for line in lines:
        x1, y1,x2, y2 = line[0]
        m, c = np.polyfit((x1, x2), (y1, y2), 1)
        if m > 0 :
            right_lane.append((m, c))

        else :
            left_lane.append((m, c))
    
    for right in right_lane:
        right_m += (right[0] / len(right_lane))
        right_c += (right[1] / len(right_lane))

    right_lane_final = [[right_m, right_c]] 

    for left in left_lane :
        left_m += (left[0] / len(left_lane))
        left_c += (left[1] / len(left_lane))

    left_lane_final = [[left_m, left_c]]

    if left_m != 0 and right_m != 0 :
        main_lane.append(get_x_y(image, left_lane_final))
        main_lane.append(get_x_y(image, right_lane_final))
        return main_lane

    else :
        return []


def track_line(image, track_point):

    steer_py, steer_px = image.shape
    blank_image = np.zeros(((image.shape[0]), image.shape[1], 3), dtype = np.uint8)
    # print(track_point)
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
        return trace

    else :
        print("no")
        return img 



img = cv.imread("track_test_image.png")
img = cv.resize(img, (640,480))

print(img.shape)
h = img.shape[0]
w = img.shape[1]
c = img.shape[2]

region_of_interest_vertices = [
    (0, 480),
    (0, 232),
    (635, 234),
    (640, 480)
]

# region_of_interest_vertices = [
#     (0, h),
#     (w/2, h/2),
#     (w, h)     
# ]

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lp = np.array([65, 72, 82])
up = np.array([166, 255, 255])
mask = cv.inRange(hsv, lp, up)
res = cv.bitwise_and(img,img,mask=mask)


# imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edge = cv.Canny(res, 100, 210)

cropped_img = region_of_interest(edge, np.array([region_of_interest_vertices], np.int32))
cv.imshow("sh", cropped_img)

lines = cv.HoughLinesP(cropped_img, 2, np.pi / 180, 100, minLineLength= 50, maxLineGap= 10)

track_point = slope_line(cropped_img, lines)

track_image = track_line(cropped_img, track_point)
cv.imshow("track", track_image)



cv.waitKey(0)
cv.destroyAllWindows()

