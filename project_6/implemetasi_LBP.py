import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Membaca gambar dalam grayscale
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_6/rock.jpg", 0)  # 0 untuk membaca gambar dalam grayscale

# Menerapkan Local Binary Pattern (LBP)
radius = 1
n_points = 8 * radius  # Menghitung jumlah titik untuk LBP
lbp = local_binary_pattern(image, n_points, radius, method='uniform')  # Menghitung LBP

# Menampilkan hasil LBP
cv2.imshow('Local Binary Patterns', lbp.astype(np.uint8))  # Menampilkan gambar LBP
cv2.waitKey(0)
cv2.destroyAllWindows()






