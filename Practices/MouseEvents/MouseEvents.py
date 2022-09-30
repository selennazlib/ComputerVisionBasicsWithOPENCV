#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2

cap = cv2.VideoCapture(0)

circles = []
def mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        circles.append((x, y))
        

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    for center in circles:
        cv2.circle(frame, center, 10, (0, 255, 0), -1)
        
    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key == ord('h'):
        circles = []
        
cap.release()
cv2.destroyAllWindows()

