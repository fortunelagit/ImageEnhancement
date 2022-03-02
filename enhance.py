import cv2
import numpy as np

## Loading source
src = cv2.imread('test.png', 1)
cv2.imshow('source', src)
src = cv2.addWeighted(src, 3, src, 0, 0)
cv2.imshow('brightened', src)

## Blurring
gray = cv2.bilateralFilter(src,9,75,75)
gray = cv2.GaussianBlur(src, (5, 5), 0)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

##Laplacian Operator
gray = cv2.Laplacian(gray, ddepth = cv2.CV_16S, ksize=7)
cv2.imshow('laplacian', gray)

##Laplacian result added to source
gray = cv2.imread('gray.png', 1)
lap = cv2.addWeighted(gray, 1.2, src, 0.7, 0)
cv2.imshow('laplacian combination', lap)

ddepth=cv2.CV_16S

##Sobel operator applied to source
grad_x = cv2.Sobel(src, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(src, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

sobel = cv2.addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0)
cv2.imshow('sobel', sobel)

##Sobel result applied with 5x5 Averaging Mask Filter
kernel = np.ones((5,5),np.float32)/25
print(kernel)
avg = cv2.filter2D(sobel,-1,kernel)
cv2.imshow('5x5 averaging', avg)

## Adding average mask result to laplacian result
mask = cv2.addWeighted(sobel, 1, lap, 0.5, 0)

## Adding result to source
mask = cv2.addWeighted(mask, 0.7, src, 0.5, 0)

## Power law transformation using gamma
mask = np.array(255*(mask/255)**1,dtype='uint8')
cv2.imshow('power law', mask)

## Add source to transformation result
mask = cv2.addWeighted(mask, 0.6, src, 0.6, 0)
cv2.imshow('final', mask)

cv2.waitKey(0)
cv2.destroyAllWindows()



