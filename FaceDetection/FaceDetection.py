#!/usr/bin/env python
# coding: utf-8

# In[16]:


# ON IMAGES
import cv2
import numpy as np

img_path = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\FaceDetection\\audrey.jpg'
cascade_path = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\FaceDetection\\frontalface.xml'

img = cv2.imread(img_path)
img = cv2.resize(img, (640, 640))
face_cascade = cv2.CascadeClassifier(cascade_path)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img, 1.2, 6)
"""
1. image : Matrix of the type CV_8U containing an image where objects are detected.
2. scaleFactor : Parameter specifying how much the image size is reduced at each image scale.
3. minNeighbors : Parameter specifying how many neighbors each candidate rectangle should have to retain it. 
This parameter will affect the quality of the detected faces: higher value results in less detections but with higher quality. 
4. flags : Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. 
It is not used for a new cascade.
5. minSize : Minimum possible object size. Objects smaller than that are ignored.
6. maxSize : Maximum possible object size. Objects larger than that are ignored.

If faces are found, it returns the positions of detected faces as Rect(x,y,w,h).
"""

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (160, 32, 240), 1)
    
    
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

