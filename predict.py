# Import libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
from PIL import Image

# OpenCV library
import cv2

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

from collections import defaultdict

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.

def showimage(image):
    if (image.ndim > 2):  # This only applies to RGB or RGBA images (e.g. not to Black and White images)
        image = image[:,:,::-1] # OpenCV follows BGR order, while matplotlib likely follows RGB order
         
    fig, ax = plt.subplots(figsize=[10,10])
    ax.imshow(image, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def remove_background(data):
    image = cv2.imread(data)
    #  Using the LAB color space. 
    # The luminance channel expressed a lot of info on the amount of brightness in the image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Obtaining a threshold and masking the result with the original image
    ret2, th = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask1 = cv2.bitwise_and(image, image, mask = th)
    
    
    # But now the background is not what you intended it to be. 
    # I created an image with white pixels of the same image dimension (white) and masked the inverted threshold image with it

    white = np.zeros_like(image)
    white = cv2.bitwise_not(white)

    mask2 = cv2.bitwise_and(white, white, mask = cv2.bitwise_not(th))
    
    kernel = np.ones((2, 2), np.uint8)
    img_dilation = cv2.dilate(mask2, kernel, iterations=3)
    result = cv2.erode(img_dilation, kernel, iterations=3) 
    
    return result

def flatten(l):
    return [item for sublist in l for item in sublist]

def intersect(c1, c2):
    x1 = c1[0]
    x2 = c2[0]
    y1 = c1[1]
    y2 = c2[1]
    w1 = c1[2]
    w2 = c2[2]
    h1 = c1[3]
    h2 = c2[3]
    if ((x1 + w1/2) < (x2 - w2/2)) or ((x1 - w1/2) > (x2 + w2/2)) or ((y1 + h1/2) < (y2 - h2/2)) or ((y1 - h1/2) > (y2 + h2/2)):
        return False
    else:
        return True
    
def overlap(c1, c2):
    x1 = c1[0]
    x2 = c2[0]
    y1 = c1[1]
    y2 = c2[1]
    w1 = c1[2]
    w2 = c2[2]
    h1 = c1[3]
    h2 = c2[3]
    if ((x1 + w1/2) <= (x2 + w2/2)) and ((x1 - w1/2) >= (x2 - w2/2)) and ((y1 + h1/2) <= (y2 + h2/2)) and ((y1 - h1/2) > (y2 - h2/2)):
        return True
    elif ((x2 + w2/2) <= (x1 + w1/2)) and ((x2 - w2/2) >= (x1 - w1/2)) and ((y2 + h2/2) <= (y1 + h1/2)) and ((y2 - h2/2) > (y1 - h1/2)):
        return True
    else:
        return False

def connected_components(lists):
    neighbors = defaultdict(set)
    seen = set()
    for each in lists:
        for item in each:
            neighbors[item].update(each)
    def component(node, neighbors=neighbors, seen=seen, see=seen.add):
        nodes = set([node])
        next_node = nodes.pop
        while nodes:
            node = next_node()
            see(node)
            nodes |= neighbors[node] - seen
            yield node
    for node in neighbors:
        if node not in seen:
            yield sorted(component(node))

def segment(removed_image):
    img_gray = cv2.cvtColor(removed_image, cv2.COLOR_BGR2GRAY)
    # apply binary thresholding
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    # visualize the binary image
    
    height, width, _ = removed_image.shape
    
    thresh = 255 - thresh
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # draw contours on the original image
    image_copy = removed_image.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    image_copy1 = removed_image.copy()
    coords = []

    for region in contours:
        x, y, w, h = cv2.boundingRect(region)
        if h <= 12.5 or w <= 12.5:
            continue
        coords.append([x, y, w, h])
        cv2.rectangle(image_copy1, (x-15, y-15), (x + w + 15, y + h + 15), (0, 255, 0), 1)
    
    merged_coords = []
    intersects = []
    overlaps = []

    for i in range(len(coords) - 1):
        for j in range(i + 1, len(coords)):
            if intersect(coords[i], coords[j]):
                intersects.append([i, j])
                continue
            if overlap(coords[i], coords[j]):
                overlaps.append([i, j])
                continue
    
    intersects = list(connected_components(intersects))
    overlaps = list(connected_components(overlaps))
    
    for i in range(len(coords)):
        if i not in flatten(intersects):
            merged_coords.append(coords[i])
    
    for c in intersects:
        min_index = 0
        x = 1000
        for j in c:
            if coords[j][0] < x:
                x = coords[j][0]
                min_index = j

        max_index = 0
        m = 10000
        for j in c:
            if coords[j][1] < m:
                m = coords[j][1]
                max_index = j

        y = coords[max_index][1]
        max_w = max((coords[i][0] + coords[i][2]) for i in c)
        w = max_w - x
        max_h = max((coords[i][1] + coords[i][3]) for i in c)
        h = max_h - y

        merged_coords.append([x, y, w, h])
        
    image_copy2 = removed_image.copy()
    alphabets = []
    merged_coords = sorted(merged_coords, key=lambda x:(-x[2] * x[3]))
    merged_coords = merged_coords[:3]
    merged_coords = sorted(merged_coords, key=lambda x:x[0])
    for coord in merged_coords:
        x,y,w,h = coord
        alphabets.append(image_copy2[y -10 : y + h + 10, x - 10 : x + w + 10])
        cv2.rectangle(image_copy2, (x-15, y-15), (x + w + 15, y + h + 15), (0, 255, 0), 1)
        
    return alphabets

def decaptcha( filenames ):
    labels = []
    names = ['ALPHA','BETA','CHI','DELTA','EPSILON','ETA','GAMMA','IOTA','KAPPA',
            'LAMDA','MU','NU','OMEGA','OMICRON','PHI','PI','PSI','RHO','SIGMA','TAU',
            'THETA','UPSILON','XI','ZETA']
    model = tf.keras.models.load_model("Model1.h5") # add model :)
    for file in filenames:
        removed_image = remove_background(file)
        alphabets = segment(removed_image)
        result = ""
        for letter in alphabets:
            try:
                letter = cv2.resize(letter,(140, 140))
                letter = np.reshape(letter, (1, 140, 140, 3))
                predict = model.predict(np.array(letter))
                result = result + str(names[np.argmax(predict)]) + ','
            except Exception as e:
                result = result + "_" + ","
            
        labels.append(result[:-1])
    return labels