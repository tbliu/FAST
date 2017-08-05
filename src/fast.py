"""
Performs FAST corner detection without machine generated code.

Reference:
    - http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html
"""

def shape(array):
    """ 
    Returns a list of 2D array dimensions 
    """
    rows = len(array)

    cols = len(array[0])
    return [rows, cols]

def rgb2gray(array):
    """
    Transforms RGB image matrix into grayscale. 
    Uses formula from: https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    """
    rows, cols = shape(array)
    for row in range(rows):
        for col in range(cols):
            red, green, blue = array[row][col]
            gray = int(0.3*red + 0.59*green + 0.11*blue)
            array[row][col] = gray


def circle(row, col):
    """ 
    Returns a list of some of the pixels ((x,y) tuples) that make up the circumference of a pixel's search region.
    Circle circumference = 16 pixels
    See: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html for details
    """
    point1 = (row+3, col)
    
    point3 = (row+3, col-1)
    
    point5 = (row+1, col+3)
    
    point7 = (row-1, col+3)
    
    point9 = (row-3, col)
    
    point11 = (row-3, col-1)
    
    point13 = (row+1, col-3)
    
    point15 = (row-1, col-3)
    
    return [point1, point3, point5, point7, point9, point11, point13, point15];

def is_corner(image, row, col, ROI, threshold):
    """
    We use a version of the high speed test (see OpenCV reference) to detect a corner:
    Uses the same pixels returned from the circle function. 
    Pixels are ordered according to the OpenCV reference (see the section titled: Feature Detection using FAST)
   
    Method:
        If the intensity on pixel 1 meets the threshold criteria, check if pixels 3 and 15 meet it as well.
        If those pixels meet the criteria, check if pixels 5 and 13 meet is as well. If so, it is a corner. 
        Repeat with every point returned from the circle function
        If none of the criteria is met, it is not a corner
        
        This way we check several points along the 12 contiguous pixel method detailed in the 
        Feature Detection Using FAST section of the OpenCV reference, which provides more accuracy
        while still maintaining the speed of the high-speed test, also detailed in the same section of the reference

    This does not reject as many candidates as checking every point in the circle,
    but it runs much faster and we can set the threshold to be a high value to filter
    out more non-corners
    """
    intensity = int(image[row][col]) # Center pixel intensity
    row1, col1 = ROI[0]
    row3, col3 = ROI[1]
    row5, col5 = ROI[2]
    row7, col7 = ROI[3]
    row9, col9 = ROI[4]
    row11, col11 = ROI[5]
    row13, col13 = ROI[6]
    row15, col15 = ROI[7]
    intensity1 = int(image[row1][col1])
    intensity3 = int(image[row3][col3])
    intensity5 = int(image[row5][col5])
    intensity7 = int(image[row7][col7])
    intensity9 = int(image[row9][col9])
    intensity11 = int(image[row11][col11])
    intensity13 = int(image[row13][col13])
    intensity15 = int(image[row15][col15])

    if abs(intensity1 - intensity) > threshold:
        if abs(intensity15 - intensity) > threshold and abs(intensity3 - intensity) > threshold:
            if abs(intensity13 - intensity) > threshold and abs(intensity5 - intensity) > threshold:
                return True
    
    if abs(intensity13 - intensity) > threshold:
        if abs(intensity15 - intensity) > threshold and abs(intensity11 - intensity) > threshold:
            if abs(intensity1 - intensity) > threshold and abs(intensity9 - intensity) > threshold:
                return True

    if abs(intensity9 - intensity) > threshold:
        if abs(intensity11 - intensity) > threshold and abs(intensity7 - intensity) > threshold:
            if abs(intensity13 - intensity) > threshold and abs(intensity5 - intensity) > threshold:
                return True

    if abs(intensity5 - intensity) > threshold:
        if abs(intensity3 - intensity) > threshold and abs(intensity7 - intensity) > threshold:
            if abs(intensity1 - intensity) > threshold and abs(intensity9 - intensity) > threshold:
                return True
    
    if abs(intensity3 - intensity) > threshold:
        if abs(intensity1 - intensity) > threshold and abs(intensity5 - intensity) > threshold:
            if abs(intensity15 - intensity) > threshold and abs(intensity7 - intensity) > threshold:
                return True

    if abs(intensity7 - intensity) > threshold:
        if abs(intensity5 - intensity) > threshold and abs(intensity9 - intensity) > threshold:
            if abs(intensity3 - intensity) > threshold and abs(intensity11 - intensity) > threshold:
                return True

    if abs(intensity11 - intensity) > threshold:
        if abs(intensity13 - intensity) > threshold and abs(intensity9 - intensity) > threshold:
            if abs(intensity7 - intensity) > threshold and abs(intensity15 - intensity) > threshold:
                return True

    if abs(intensity15 - intensity) > threshold:
        if abs(intensity13 - intensity) > threshold and abs(intensity1 - intensity) > threshold:
            if abs(intensity11 - intensity) > threshold and abs(intensity3 - intensity) > threshold:
                return True

    return False

def areAdjacent(point1, point2):
    """
    Identifies if two points are adjacent by calculating distance in terms of rows/cols
    Two points are adjacent if they are within four pixels of each other (Euclidean distance)
    """
    row1, col1 = point1
    row2, col2 = point2
    xDist = row1 - row2
    yDist = col1 - col2
    return pow(pow(xDist, 2) + pow(yDist, 2), 0.5) <= 4

def calculateScore(image, point):
    """ 
    Calculates the score for non-maximal suppression. 
    The score V is defined as the sum of the absolute difference between the intensities of 
    all points returned by the circle function and the intensity of the center pixel.
    """
    row, col = point
    intensity = int(image[row][col])
    
    intensity1 = image[row+3][col] 
    
    intensity3 = image[row+3][col-1] 
    
    intensity5 = image[row+1][col+3]    
    
    intensity7 = image[row-1][col+3]
    
    intensity9 = image[row-3][col] 
    
    intensity11 = image[row-3][col-1] 
    
    intensity13 = image[row+1][col-3] 
    
    intensity15 = image[row-1][col-3]
       
    score = abs(intensity - intensity1) + abs(intensity - intensity3) + \
            abs(intensity - intensity5) + abs(intensity - intensity7) + \
            abs(intensity - intensity9) + abs(intensity - intensity11) + \
            abs(intensity - intensity13) + abs(intensity - intensity15)
    return score


def suppress(image, corners):
    """
    Performs non-maximal suppression on the list of corners.
    For adjacent corners, discard the one with the smallest score.
    Otherwise do nothing

    Since we iterate through all the pixels in the image in order, any adjacent 
    corner points should be next to each other in the list of all corners

    Non-maximal suppression throws away adjacent corners which are the same point in real life
    """
    i = 1
    while i < len(corners):
        currPoint = corners[i]
        prevPoint = corners[i - 1]
        if areAdjacent(prevPoint, currPoint):
            currScore = calculateScore(image, currPoint)
            prevScore = calculateScore(image, prevPoint)
            if (currScore > prevScore):
                del(corners[i - 1])
                i += 1
            else:
                del(corners[i])
        else:
            i += 1
            continue
    return

def detect(image, threshold=100):
    """
    corners = fast.detect(image, threshold) performs the detection
    on the image and returns the corners as a list of (x,y) tuples
    where x is the column index, and y is the row index

    Nonmaximal suppression is implemented by default. 

    This function does not search the entire frame for corners. It only searches a portion
    in the middle in order to speed up the process.

    ***Parameters: 
        image is a numpy array of intensity values. NOTE: Image must be grayscale
        threshold is an int used to filter out non-corners. 
    """

    corners = []
    imshape = shape(image)
    rows = imshape[0]
    cols = imshape[1]
    startSearchRow = int(0.25*rows)
    endSearchRow = int(0.75*rows) # search the middle square of the frame
    startSearchCol = int(0.25*cols)
    endSearchCol = int(0.75*cols)
    for row in range(startSearchRow, endSearchRow):
        for col in range(startSearchCol, endSearchCol):
            ROI = circle(row, col) 
            if is_corner(image, row, col, ROI, threshold):
                corners.append((col, row))
    #suppress(image, corners) 
    return corners;
