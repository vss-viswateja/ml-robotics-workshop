# importing libraries 
import cv2 
import numpy as np 

  
image_1 = cv2.imread('level-4/datasets/gabriel-martin-FH3NzSqwOTU-unsplash.jpg') 
image_2 = cv2.imread('level-4/datasets/awmleer-I--YyrXUphc-unsplash.jpg') 
image_3 = cv2.imread('level-4/datasets/daniel-lloyd-blunk-fernandez-QkKKggRWlE8-unsplash.jpg') 
  


#resize
new_img_1 = cv2.resize(image_1,(128,128))
new_img_2 = cv2.resize(image_2,(128,128))
new_img_3 = cv2.resize(image_3,(128,128))


cv2.imshow('Original Image', new_img_1) 
cv2.waitKey(0) 

cv2.imshow('Original Image', new_img_2) 
cv2.waitKey(0) 

cv2.imshow('Original Image', new_img_3) 
cv2.waitKey(0) 

# Gaussian Blur 
Gaussian = cv2.GaussianBlur(new_img_1, (7, 7), 0) 
cv2.imshow('Gaussian Blurring', Gaussian) 
cv2.waitKey(0) 
# Gaussian Blur 
Gaussian = cv2.GaussianBlur(new_img_2, (7, 7), 0) 
cv2.imshow('Gaussian Blurring', Gaussian) 
cv2.waitKey(0) 
# Gaussian Blur 
Gaussian = cv2.GaussianBlur(new_img_3, (7, 7), 0) 
cv2.imshow('Gaussian Blurring', Gaussian) 
cv2.waitKey(0) 
'''
Horse : 255,575

Dog: 125, 750


Human 300,750




'''