#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2

# webcam videos
cap = cv2.VideoCapture(0) # if we use our computer's webcam we put 0 as a parameter otherwise give the path of the video

fileName = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\CV2VideoBasics\\webcam.avi'
codec = cv2.VideoWriter_fourcc('X','V','I','D') #  Windows Media Video 8 codec.
frameRate = 30
resolution = (640, 180)
videoFileOutput = cv2.VideoWriter(fileName, codec, frameRate, resolution)

while True:
    
    ret, frame = cap.read() # if reads correct ret = true
    frame = cv2.flip(frame, 1) # to rotate the frame 
    
    if ret == 0:
        break # this if statement for the videos that we downloaded and it helps to break when the video finished
        
    cv2.imshow("webcam", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break # show the each frame 30 milisecond and quit when press q

videoFileOutput.release()
cap.release()
videoFileOutput.release()
cv2.destroyAllWindows()


# In[ ]:




