#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import requests

url = 'http://192.168.0.148:8080//shot.jpg' # generated this url with ip webcam app that I downloadded my phone
while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 480))
    
    cv2.imshow('Android Cam', img)
    if cv2.waitKey(1) == 27:
        break
        
cv2.destroyAllWindows()


# In[ ]:




