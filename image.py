"""
Triple qouted text is the description,

lines starting with # are comments
"""

"""importing the opencv package"""
import cv2

"""
reading an image
"""
img = cv2.imread('lena_color_512.tif') #read the image file and load into img variable

"""
creating a copy of the image
"""
face_img = img.copy()


"""loading the trained algorithm"""

algo = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


"""
detecting faces
"""
faces = algo.detectMultiScale(img)

"""
drawing rectangles around detected faces
"""
for x,y,w,h in faces:
    cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,255,0),3)


"""
view the image
"""
cv2.imshow('original',img)
cv2.imshow('face detected',face_img)
cv2.waitKey(0)
cv2.destroyAllWindows()








