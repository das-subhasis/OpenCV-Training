import cv2  
import os

IMG_DIR = 'dataset/train/cat/0.jpg'

img  = cv2.imread(IMG_DIR,1)  

retval, threshold = cv2.threshold(img, 152, 255, cv2.THRESH_BINARY)  

cv2.imshow("Original Image", img)  
cv2.imshow("Threshold",threshold)  

cv2.waitKey(0)  