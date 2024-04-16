import cv2
import numpy as np
import os


IMG_DIR = 'dataset/train/cat/0.jpg'

image = cv2.imread(IMG_DIR, 1)

edges = cv2.Canny(image, 100,200)

cv2.imshow('image', edges)

cv2.waitKey(0)

cv2.destroyAllWindows()