import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    print(x)

cv.namedWindow("track")    

img = cv.imread("stuff.jpg", 0)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

lap = cv.Laplacian(img, cv.CV_64F, ksize=3)#64f is s4bit float that deals with -ve no n this is due to image is converted from white to black that creates a -ve slpoe
lap = np.uint8(np.absolute(lap))#returns the absolute no in 8 bit

sx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)# intensity increases vertically
sx = np.uint8(np.absolute(sx))#returns the absolute no in 8 bit

sy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)# intensity increases horizontally
sy = np.uint8(np.absolute(sy))#returns the absolute no in 8 bit

sxorsy = cv.bitwise_or(sx,sy)

cny = cv.Canny(img, 110, 255)

cv.createTrackbar("Th1", "track", 120, 255, nothing)
cv.createTrackbar("Th2", "track", 200, 255, nothing)


titles = ["image", "laplascian", "sx", "sy", "sx,sy", "canny"]
images = [img, lap, sx, sy, sxorsy, cny]

th1 = cv.getTrackbarPos("TH1", "track")
th2 = cv.getTrackbarPos("TH2", "track")
# cny = cv.Canny(img, th1, th2)

for i in range (0, 6):
    plt.subplot(2, 3, i+1)
    plt.title(titles[i])
    plt.imshow(images[i],"gray")
    plt.xticks([]), plt.yticks([])

plt.show()    
cv.destroyAllWindows()