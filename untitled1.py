# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:41:58 2021

@author: HP
"""

import cv2
import tensorflow.keras
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model1.h5')

"""
opening a video
"""
cap = cv2.VideoCapture(0) # 0 will open the video from default web cam

"""loading the trained algorithm to detect faces""" 

algo = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels = ["Subham","Satyam"]

def get_predict(img,model,labels):
    img = cv2.resize(frame,(224,224))
    img = img.reshape(1, 224, 224, 3)
    img = np.asarray(img,dtype=np.float32)
    
    # Normalize the image
    normalized_image_array = (img / 127.0) - 1
    # run the inference
    prediction = model.predict(normalized_image_array)
    print(prediction)
    max_pred = prediction.max()
    print(max_pred)
    i = np.where(prediction == max_pred)
    if max_pred>0.98:
        return labels[i[0][0]]
    else :
        return None
    
"""
starting the video loop
"""

while True:
    
    ret,frame = cap.read() # reading the video stream
    
    faces = algo.detectMultiScale(frame)
    
    if faces is not None:
        
        for x,y,w,h in faces:
            face_img = frame[y-2:y+(h+2),x-2:x+(w+2)]
            name = get_predict(face_img, model, labels)
            if name is not None:
                cv2.rectangle(frame,(x,y),(x+w,y+h),
                              (0,255,0),3)
                cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),
                              (0,0,255),3)
                cv2.putText(frame,"unknown",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)                
        
    cv2.imshow('video',frame)
    
    if cv2.waitKey(30)==ord('q'): #press 'q' to quit
        break
    
cap.release() #release the camera
cv2.destroyAllWindows()