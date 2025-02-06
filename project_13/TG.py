import cv2
import numpy as np 

#baca citra dalam grayscale 
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_13/punk.jpg", cv2.IMREAD_GRAYSCALE)

#terapkan tresholding global 
ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

#tampilkan hasil segmentasi 
cv2.imshow('Thresholdwd Image', thresh)
cv2.waitKey(0)
cv2.destroyWindow()

