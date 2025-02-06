import cv2
import numpy as np

#baca citra Grayscale
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_11/punk.jpg", cv2.IMREAD_GRAYSCALE)

#Deteksi tepi Menggunakan Sobel
Sobelx = cv2.Sobel(image,cv2. CV_64F, 1, 0, ksize=5) #sobel X
Sobely = cv2.Sobel(image,cv2.CV_64F, 0, 1, ksize=5) #Sobel Y

# Tampilkan Hasil
cv2.imshow('Sobel X', Sobelx)
cv2.imshow('Sobel Y', Sobely)
cv2.waitKey(0)
cv2.destroyAllWindows() 