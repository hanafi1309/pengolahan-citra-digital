import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Baca citra dalam grayscale
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_14/punk.jpg", cv2.IMREAD_GRAYSCALE)

# Terapkan Local Binary Pattern
radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(image, n_points, radius, method='uniform')

# Konversi LBP ke tipe data uint8 dan tampilkan hasil
lbp = lbp.astype(np.uint8)
cv2.imshow('Local Binary Pattern', lbp)
cv2.waitKey(0)
cv2.destroyAllWindows()
