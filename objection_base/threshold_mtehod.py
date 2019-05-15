import numpy as np
import cv2


if __name__ == '__main__':
    # 1、读取图片
    img = cv2.imread('anwser_sheet.jpg')
    # 2、将图片转换为灰度图像
    grey = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 对图像进行阈值处理，
    retval, grey = cv2.threshold(grey, 90, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('after_threshold', grey)
    cv2.waitKey()

    grey = cv2.erode(grey, None)
    cv2.imshow('after_erode', grey)
    cv2.waitKey()
    grey = cv2.dilate(grey, None)
    cv2.imshow('after_dilate', grey)
    cv2.waitKey()

    contours, hierarchy = cv2.findContours(grey.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    newimg = np.zeros_like(grey)
    cv2.drawContours(newimg, contours, -1, 255)
    # cv2.drawContours(newimg, contours, 0, 255)
    cv2.imshow('test', newimg)
    cv2.imwrite("processed.jpg", newimg)
    cv2.waitKey()