import cv2
import numpy as np
# Membaca gambar
image = cv2.imread('c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_4/rock.jpg', 0)

kernel = np.ones((5, 5), np.uint8)

dilated_image = cv2.dilate(image, kernel, iterations=1)
eroded_image = cv2.erode(image, kernel, iterations=1)

cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()