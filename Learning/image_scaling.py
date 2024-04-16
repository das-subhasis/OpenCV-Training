import cv2
import numpy as np
import os


IMG_DIR = 'dataset/train/cat/0.jpg'

image = cv2.imread(IMG_DIR,1)

scale = 200

height = int(image.shape[0] * scale / 100)
width = int(image.shape[1] * scale / 100)

image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

cv2.imshow('img',image)

cv2.waitKey(2000)
