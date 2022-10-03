#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2

img_path = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\EyeDetection\\audrey.jpg'
eye_cascade_path = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\EyeDetection\\eye.xml'
face_cascade_path = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\EyeDetection\\frontalface.xml'

img = cv2.imread(img_path)
img = cv2.resize(img, (740, 640))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
face_cascade = cv2.CascadeClassifier(face_cascade_path)

faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 5, 0), 1)


img2 = img[y: y + h, x: x + w]
gray2 = gray_img[y: y + h, x: x + w]

eyes = eye_cascade.detectMultiScale(gray2)

for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(img2, (ex, ey), (ex + ew, ey + eh), (55, 85, 0), 1)
    
cv2.imshow('Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

