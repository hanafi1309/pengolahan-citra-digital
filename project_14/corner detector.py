import cv2
import numpy as np

# Baca citra asli
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_14/punk.jpg")
if image is None:
    print("Error: File 'punk.jpg' tidak ditemukan.")
    exit()

# Konversi ke grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Terapkan Harris Corner Detector
gray = np.float32(gray)
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# Tingkatkan sudut yang terdeteksi (gunakan mask boolean untuk menandai sudut pada citra berwarna)
image[corners > 0.01 * corners.max()] = [0, 0, 255]

# Tampilkan hasil deteksi sudut
cv2.imshow('Harris Corner Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
