# -*- coding:utf-8 -*-

import numpy as np
import cv2.cv2 as cv


''' 
函数名称：calc_hist
功能：计算直方图
输入：
mag    幅值矩阵
angle  角度矩阵，范围在 0-180
bin_size    直方图区间大小
输出：
hist    直方图
'''
def calc_hist(mag, angle, bin_size=9):
    hist = np.zeros((bin_size,), dtype=np.int32)

    bin_step = 180 // bin_size
    bins = (angle // bin_step).flatten()
    flat_mag = mag.flatten()

    for i,m in zip(bins, flat_mag):
        hist[i] += m

    return hist


# 归一化cells
def l2_norm(cells):
    block = cells.flatten().astype(np.float32)
    norm_factor = np.sqrt(np.sum(block**2) + 1e-6)
    block /= norm_factor
    return block


# 计算HOG特征
def calc_hog(gray):
    ''' 计算梯度 '''
    dx = cv.Sobel(gray, cv.CV_16S, 1, 0)
    dy = cv.Sobel(gray, cv.CV_16S, 0, 1)
    sigma = 1e-3
    # 计算角度
    angle = np.int32(np.arctan(dy / (dx + sigma)) * 180 / np.pi) + 90
    dx = cv.convertScaleAbs(dx)
    dy = cv.convertScaleAbs(dy)
    # 计算梯度大小
    mag = cv.addWeighted(dx, 0.5, dy, 0.5, 0)

    print('angle\n', angle[:8,:8])
    print('mag\n', mag[:8,:8])
    ''' end of 计算梯度 '''

    # 将图像切成多个cell
    cell_size = 8
    bin_size = 9
    img_h, img_w = gray.shape[:2]
    cell_h, cell_w = (img_h // cell_size, img_w // cell_size)

    cells = np.zeros((cell_h, cell_w, bin_size), dtype=np.int32)
    for i in range(cell_h):
        cell_row = cell_size * i
        for j in range(cell_w):
            cell_col = cell_size * j
            cells[i,j] = calc_hist(mag[cell_row:cell_row+cell_size, cell_col:cell_col+cell_size],
                angle[cell_row:cell_row+cell_size, cell_col:cell_col+cell_size], bin_size)

    # 多个cell融合成block
    block_size = 2
    block_h, block_w = (cell_h-block_size+1, cell_w-block_size+1)
    blocks = np.zeros((block_h, block_w, block_size*block_size*bin_size), dtype=np.float32)
    for i in range(block_h):
        for j in range(block_w):
            blocks[i,j] = l2_norm(cells[i:i+block_size, j:j+block_size])

    return blocks.flatten()


if __name__ == '__main__':
    pass