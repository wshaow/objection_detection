import cv2.cv2 as cv
import numpy as np
import os

if __name__ == '__main__':
    img = cv.imread('lines.png')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 75, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv.imshow('ori_image', img)
    cv.imshow('gray_image', gray)
    cv.imshow('canny', edges)

    cv.waitKey()
    cv.destroyAllWindows()



