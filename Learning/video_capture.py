import numpy as np  
import cv2  
  
cap = cv2.VideoCapture(0)  
  
while(cap.isOpened()):

    ret, frame = cap.read()

    cv2.imshow('Video',frame)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()