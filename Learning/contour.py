import numpy as np  
import cv2 as cv  

IMG_DIR = 'dataset/train/cat/0.jpg'

img = cv.imread(IMG_DIR)  

imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  

ret, thresh = cv.threshold(imgray, 127, 255, 0)  

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  