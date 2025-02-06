import cv2
import numpy as np

#baca citra digital 
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_9/punk.jpg", cv2.IMREAD_GRAYSCALE)

#global Thresholdinng
ret, thresh1 = cv2.threshold (image, 127, 255, cv2.THRESH_BINARY)

#Adaptif thresholding
thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


#Tampilkan Hasil
cv2.imshow ('Global Tresholding', thresh1)
cv2.imshow ('Global Tresholding', thresh2)
cv2.waitKey(0)
cv2.destroyallWindows()