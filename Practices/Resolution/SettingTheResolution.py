#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2

cv2.namedWindow('Live video')

cap = cv2.VideoCapture(0)
# print('Width: ', str(cap.get(3))) # 640
# print('Height: ', str(cap.get(4))) # 480

cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    cv2.imshow('Live video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

