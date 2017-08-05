from fast import *
from matplotlib import pyplot as plt
import cv2
import numpy as np
def test():
    #image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/ansel.jpg');
    image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/noisy.png');
    #image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/chessboard.jpg');
    image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/balloons_noisy.png');
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #rgb2gray(image)
    kernel = np.ones((5,5),np.float32)/25
    cv2.filter2D(image,-1,kernel) 
    corners = detect(image)
    implot = plt.imshow(image, cmap='gray')
    for point in corners:
        plt.scatter(point[0], point[1], s=10)
    plt.show()

test()
