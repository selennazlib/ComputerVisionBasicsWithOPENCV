#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# HAPPY 
import cv2

img_path = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\SmileDetection\\smile.jpg'
smile_cascade_path = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\SmileDetection\\smile.xml'
face_cascade_path = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\SmileDetection\\frontalface.xml'

img = cv2.imread(img_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
face_cascade = cv2.CascadeClassifier(face_cascade_path)

img = cv2.resize(img, (640, 640))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (250, 204, 0), 3)


roi_img = img[y: y + h, x: x + w]
roi_gray = gray[y: y + h, x: x + w]

smiles = smile_cascade.detectMultiScale(roi_gray, 1.9, 7)

for (sx, sy, sw, sh) in smiles:
    cv2.rectangle(roi_img, (sx, sy), (sx + sw, sy + sh), (250, 0, 104), 2)
    
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


# UPSET

img_path = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\SmileDetection\\upset.jpg'
img = cv2.imread(img_path)

img = cv2.resize(img, (640, 640))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (250, 204, 0), 3)


roi_img = img[y: y + h, x: x + w]
roi_gray = gray[y: y + h, x: x + w]

smiles = smile_cascade.detectMultiScale(roi_gray, 1.9, 7)

for (sx, sy, sw, sh) in smiles:
    cv2.rectangle(roi_img, (sx, sy), (sx + sw, sy + sh), (250, 0, 104), 2)
    
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

