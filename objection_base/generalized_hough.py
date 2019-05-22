# -*- coding=utf-8 -*-
__author__ = 'wshaow'

import numpy as np
import cv2.cv2 as cv
from collections import defaultdict


def buildRefTable(img):
    """
    builds the reference table for the given input template image
    :param im: input binary image
    :return:
        table = a reconstructed reference table...
    """
    table = defaultdict(list)  # [[0 for x in range(1)] for y in range(90)]  # creating a empty list
    # r will be calculated corresponding to this point
    img_center = [img.shape[0]/2, img.shape[1]/2]

    dx = cv.Sobel(img, -1, 1, 0)
    dy = cv.Sobel(img, -1, 0, 1)

    gradient = np.arctan2(dy, dx) * 180 / np.pi

    for (i, j), value in np.ndenumerate(img):
        if value:
            table[gradient[i, j]].append((img_center[0] - i, img_center[1] - j))

    return table


def matchTable(im, table):
    """
    :param im: input binary image, for searching template
    :param table: table for template
    :return:
        accumulator with searched votes
    """
    # matches the reference table with the given input
    # image for testing generalized Hough Transform
    m, n = im.shape
    acc = np.zeros((m, n))  # acc array requires some extra space

    #  这里是用梯度来作为key， 然后r作为后面的value这样确实是比较说的通的做法！！！！
    dx = cv.Sobel(im, -1, 1, 0)
    dy = cv.Sobel(im, -1, 0, 1)
    gradient = np.arctan2(dy, dx) * 180 / np.pi

    for (i,j),value in np.ndenumerate(im):
        if value:
            for r in table[gradient[i,j]]:
                accum_i, accum_j = i+r[0], j+r[1]
                if accum_i < acc.shape[0] and accum_j < acc.shape[1]:
                    acc[int(accum_i), int(accum_j)] += 1

    return acc


def findMaxima(acc):
    """
    :param acc: accumulator array
    :return:
        maxval: maximum value found
        ridx: row index of the maxval
        cidx: column index of the maxval
    """
    ridx, cidx = np.unravel_index(acc.argmax(), acc.shape)
    return [acc[ridx, cidx], ridx, cidx]


if __name__ == '__main__':
    images = [r'ghough/Input1Ref.png', r'ghough/Input2Ref.png']
    for img in images:
        refim = cv.imread(img, cv.COLOR_BGR2BGRA)
        im = cv.imread(r'ghough/Input1.png', cv.COLOR_BGR2BGRA)
        cv.imshow(img, refim)
        cv.imshow('Input1', im)
        table = buildRefTable(refim)
        acc = matchTable(im, table)
        val, ridx, cidx = findMaxima(acc)

        # code for drawing bounding-box in accumulator array...

        acc[ridx - 5:ridx + 5, cidx - 5] = val
        acc[ridx - 5:ridx + 5, cidx + 5] = val

        acc[ridx - 5, cidx - 5:cidx + 5] = val
        acc[ridx + 5, cidx - 5:cidx + 5] = val

        cv.imshow('para space', acc)
        # code for drawing bounding-box in original image at the found location...

        # find the half-width and height of template
        hheight = np.floor(refim.shape[0] / 2) + 1
        hwidth = np.floor(refim.shape[1] / 2) + 1

        # find coordinates of the box
        rstart = int(max(ridx - hheight, 1))
        rend = int(min(ridx + hheight, im.shape[0] - 1))
        cstart = int(max(cidx - hwidth, 1))
        cend = int(min(cidx + hwidth, im.shape[1] - 1))

        # draw the box
        im[rstart:rend, cstart] = 255
        im[rstart:rend, cend] = 255

        im[rstart, cstart:cend] = 255
        im[rend, cstart:cend] = 255

        # show the image
        cv.imshow('reference image', refim)
        cv.imshow('feature image', im)
        cv.waitKey()
