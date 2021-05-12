# -*- coding: utf-8 -*-
"""
Created on Sat May  8 12:35:52 2021

@author: shamaun
"""

import cv2

"""
opening a video
"""
cap = cv2.VideoCapture(0) # 0 will open the video from default web cam

"""loading the trained algorithm"""

algo = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

"""
starting the video loop
"""

while True:
    
    ret,frame = cap.read() # reading the video stream
    
    faces = algo.detectMultiScale(frame)
    
    if faces is not None:
        
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),
                          (0,0,255),3)
        
    cv2.imshow('video',frame)
    
    if cv2.waitKey(30)==ord('q'): #press 'q' to quit
        break
    
cap.release() #release the camera
cv2.destroyAllWindows()
        





