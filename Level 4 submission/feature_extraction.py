import cv2 as cv
import numpy as np 
#import images as grey scale and resize them to 128x128 
c_img_1 = cv.resize(cv.imread('level-4/datasets/awmleer-I--YyrXUphc-unsplash.jpg'),(520,520))
c_img_2 = cv.resize(cv.imread('level-4/datasets/daniel-lloyd-blunk-fernandez-QkKKggRWlE8-unsplash.jpg'),(520,520))
c_img_3 = cv.resize(cv.imread('level-4/datasets/gabriel-martin-FH3NzSqwOTU-unsplash.jpg'),(520,520))

img_1 = cv.cvtColor( c_img_1,cv.COLOR_BGR2GRAY )
img_2 = cv.cvtColor( c_img_2,cv.COLOR_BGR2GRAY )
img_3 = cv.cvtColor( c_img_3,cv.COLOR_BGR2GRAY )


cv.imshow('original image ', img_1)
cv.waitKey(0)

cv.imshow('original image ', img_2)
cv.waitKey(0)

cv.imshow('original image ', img_3)
cv.waitKey(0)

#perfrom Canny edge detection
e_img_1 = cv.Canny(img_1,150,150)
e_img_2 = cv.Canny(img_2,150,550)
e_img_3 = cv.Canny(img_3,100,300)


cv.imshow('Detected edges ', e_img_1)
cv.waitKey(0)

cv.imshow('Detected edges ', e_img_2)
cv.waitKey(0)

cv.imshow('Detected edges ', e_img_3)
cv.waitKey(0)




#convert the input image into float32 type
f_img_1 = np.float32(img_1)
f_img_2 = np.float32(img_2)
f_img_3 = np.float32(img_3)

#detection of corners using harris method

rscores_1 = cv.cornerHarris(f_img_1,2,3,0.04)
rscores_1 = cv.dilate(rscores_1,None)
threshold_1 = 0.05*rscores_1.max()              # Threshold value can be changed according to the image and conditions.
c_img_1[rscores_1>threshold_1] = [0,255,0]    # Corenrs now green Color.

cv.imshow('dst',c_img_1)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()


#detection of corners using harris method

rscores_2 = cv.cornerHarris(f_img_2,2,3,0.04)
rscores_2 = cv.dilate(rscores_2,None)
threshold_2 = 0.05*rscores_2.max()              # Threshold value can be changed according to the image and conditions.
c_img_2[rscores_2>threshold_2] = [0,255,0]    # Corenrs now green Color.

cv.imshow('dst',c_img_2)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()


#detection of corners using harris method

rscores_3 = cv.cornerHarris(f_img_3,2,3,0.04)
rscores_3 = cv.dilate(rscores_3,None)
threshold_3 = 0.05*rscores_3.max()              # Threshold value can be changed according to the image and conditions.
c_img_3[rscores_3>threshold_3] = [0,255,0]    # Corenrs now green Color.

cv.imshow('dst',c_img_3)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()

