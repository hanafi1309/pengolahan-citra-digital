import cv2
import numpy as np

#membaca gambar dalam grayscle
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_5/rock.jpg",0)

#menerapkan thersholding
ret, thresh_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

#menampilkan hasil 
cv2.imshow('Thresholded Image', thresh_image)
cv2.waitKey(0)  
cv2.destroyAllWindows()
