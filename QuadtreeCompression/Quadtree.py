import numpy as np
import cv2
import sys


def quadtree(img,thresh):
    quadtree_rec(img)


def quadtree_rec(img,thresh):
    if len(img) <= 2 or len(img[0]) <= 2:
        return
    else:
        img_arr = split(img)  # split img to 4 subimage
        large_idx = 0 # largest error index
        large_error = 0 # largest error value
        for i, im in enumerate(img_arr):
            allval = im[im > -1]
            avg = sum(allval) / len(allval)
            error = np.sum(np.abs(np.subtract(allval, avg))) / len(allval)
            if error > large_error:
                large_error = error
                large_idx = i
        if large_error <=thresh:
            return
        quadtree_rec(img_arr[large_idx])  # recusively call on largest error image


def split(img):
    s_r = len(img) // 2  # center of row
    s_c = len(img[0]) // 2  # center of col
    return [img[:s_r, :s_c], img[s_r:, s_c:], img[s_r:, s_c:], img[s_r:, :s_c], img[:s_r, s_c:]]


if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])

    img_gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # turn image to gray
    quadtree(img_gray1,int(sys.argv[2]))
