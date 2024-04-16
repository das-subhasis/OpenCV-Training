import cv2
import numpy as np
import os


IMG_DIR = 'dataset/train/cat/0.jpg'

image = cv2.imread(IMG_DIR,1)

# Gets the BGR value of the image
pixel = image[100, 100]


# print(pixel)
# Displays the image in a new window
# cv2.imshow('image',image)

# Delays the window to close
# cv2.waitKey(500)


print(image.shape) # image height, width, channels

