import cv2
import numpy as np

IMG_DIR = 'dataset/train/cat/4.jpg'

image = cv2.imread(IMG_DIR)

imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

template = cv2.imread('dataset/template/4.jpg',0)

w,h =template.shape

res = cv2.matchTemplate(imgray,template,cv2.TM_CCOEFF_NORMED)

threshold = 0.8

loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,255,255),2)

cv2.imshow('cat', image)

cv2.waitKey(0)

cv2.destroyAllWindows()