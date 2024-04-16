import os
import cv2


IMG_DIR = 'dataset/train/cat/3.jpg'

image = cv2.imread(IMG_DIR)


filter = cv2.bilateralFilter(image, 9, 20, 20)


cv2.imshow('image',filter)

cv2.waitKey(0)

cv2.destroyAllWindows()