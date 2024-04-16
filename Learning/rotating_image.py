import cv2
import numpy as np
import os


IMG_DIR = 'dataset/train/cat/0.jpg'

image = cv2.imread(IMG_DIR, 1)


# get the dimensions
(height, width) = image.shape[:2]

center = (height/2, width/2)

scale = 1.0

M = cv2.getRotationMatrix2D(center=center, angle=90,scale=1.5)

rotated90 = cv2.warpAffine(image, M, (height, width))

cv2.imshow('img', rotated90)

cv2.waitKey(2000)
