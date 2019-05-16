import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img = cv.imread('dave.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

