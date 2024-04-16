import cv2
import numpy as np
import os


IMG_DIR = 'dataset/train/cat/0.jpg'

image = cv2.imread(IMG_DIR, 1)

blur = cv2.blur(image, (3,3))

cv2.imshow('Original image', image)
cv2.imshow('blurred image', blur)
cv2.imshow('Median blurred image', cv2.medianBlur(image, 5))

cv2.waitKey(0)

cv2.destroyAllWindows()