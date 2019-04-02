import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

# return unique count of each color value
def histogram(img):
    return np.unique(img, return_counts=True)

# find upper bound of image and draw image
def bound(img):
    h = histogram(img)

    plt.plot(h[0][:], h[1][:])
    plt.title("histogram of image")
    plt.savefig("histogram.jpg")
    zero = h[0][np.argmax(h[1])]

    upper = h[0][np.argmax(h[1])]
    c = np.delete(h[1], np.argmax(h[1]))
    lower = h[0][np.argmax(c)]

    return upper, lower, zero

# actual Otsu method
def segment(img):
    upper,lower,zero = bound(img)

    T = (int(upper) + int(lower)) / 2
    mask = img.copy()
    min = 10000000000
    total = len(mask[mask>-1])
    for i in range(1,256):
        left = mask[mask > i]
        left_count = len(mask[mask > i])
        right = mask[mask <= i]
        right_count = len(mask[mask <= i])
        if left_count == 0 or right_count==0:
            continue
        avg = sum(mask[mask > i])/left_count
        avg1 = sum(mask[mask <= i]) / right_count
        var = sum(np.power(np.subtract(left,avg),2))/left_count

        var1 = sum(np.power(np.subtract(right,avg1),2))/right_count

        if var*left_count/total + var1*right_count/total< min:
            T = i
            min = var*left_count/total + var1*right_count/total
    print(T)
    mask[mask > T] = 255

    mask[mask <= T] = 0
    cv2.imwrite("process.jpg",mask)

if __name__ == '__main__':

    img = cv2.imread(sys.argv[1])

    img_gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # turn image to gray
    segment(img_gray1)