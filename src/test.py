from fast import *
from matplotlib import pyplot as plt
import cv2
import numpy as np
def test():
    #image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/noisy.png')
    image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/balloons_noisy.png')
    #image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/lena.png')
    imgray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    imgray = cv2.medianBlur(imgray, 7)
    corners = detect(image)
    implot = plt.imshow(imgray, cmap='gray')
    for point in corners:
        plt.scatter(point[0], point[1], s=10)
    plt.show()

def testMedianBlur():
    #image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/lena.png')
    image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/balloons_noisy.png')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rows, cols = shape(image)
    startSearchRow = int(0.25*rows)
    endSearchRow = int(0.75*rows)
    startSearchCol = int(0.25*cols)
    endSearchCol = int(0.75*cols)
    image = medianBlur(image, startSearchRow, endSearchRow, startSearchCol, endSearchCol)
    implot = plt.imshow(image, cmap='gray')
    plt.show()

def testgray():
    image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/balloons_noisy.png')
    image = rgb2gray(image)
    print(image[0][0])
    implot = plt.imshow(image, cmap='gray')
    plt.show()

def testInsertionSort():
    lst = [3,4,12,16,1, 0]
    insertionSort(lst)
    print(lst)

#testInsertionSort()
#testgray()
#testMedianBlur()
test()
