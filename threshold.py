import cv2 as cv

img = cv. imread("gradient.png", 0)
img = cv.resize(img,(512,512))
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
status, thr = cv.threshold(img,127,255,cv.THRESH_BINARY)
status, thr1 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
status, thr2 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
status, thr3 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
status, thr4 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
status, thr5 = cv.threshold(img,127,255,cv.THRESH_BINARY)
athr = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,5,0)#5 is the no.of blocks to be t7alen at a time n the greater the vaue of cthe lighter the image will become
athr1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,5,-1)

cv.imshow("gr",thr4)
# cv.imshow("gr1",thr1)
# cv.imshow("gr2",thr2)
# cv.imshow("gr3",thr3)
# cv.imshow("gr5",thr5)
# cv.imshow("gr4",thr4)
# cv.imshow("gr6",athr)
# cv.imshow("gr7",athr1)

cv.waitKey(0)
cv.destroyAllWindows()