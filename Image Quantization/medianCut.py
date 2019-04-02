import cv2
import numpy as np
import sys

ans = [] # final answer

def medianCut(bit_depth,img,outfile,featureNum):
    mapColor = {}   # map color value to quantified value
    w,h,c = img.shape
    img_1d = img.reshape((-1,featureNum))
    img_1d_T = img_1d.T

    longest(img_1d_T,img_1d,bit_depth)
    for a in ans:
        ave = np.average(a,0)
        for r,g,b in a:
            mapColor[(r,g,b)] = ave

    for x in range(w):  #plot back to actual image
        for y in range(h):
            r,g,b = img[x,y]
            img[x,y] = mapColor[(r,g,b)]
    cv2.imwrite(outfile+".jpg",img)


# find idx with longest difference and recursively call longest to cut further
def longest(img,img_1d,depth):
    if depth!=0:
        max_dim = np.argmax(np.max(img,axis=1)-np.min(img,axis=1))
        img_id_sorted = img_1d[np.argsort(img_1d[:, max_dim])]

        splitIdx = len(img_id_sorted) // 2
        left, right = img_id_sorted[:splitIdx], img_id_sorted[splitIdx:]

        longest(left.T,left,depth-1)
        longest(right.T,right,depth-1)
    else:
        ans.append(img_1d)






if __name__ == '__main__':
    filename = sys.argv[1]
    bit_depth = int(sys.argv[2])
    outfile = sys.argv[3]
    img = cv2.imread(filename)
    medianCut(bit_depth,img,outfile,featureNum=3)
