import cv2
import numpy as np

# Baca citra
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_13/punk.jpg")

# Konversi citra ke grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding untuk mendapatkan citra biner
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Penghapusan noise menggunakan Morphological Opening
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Tentukan area latar belakang
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Tentukan area objek menggunakan Distance Transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Temukan area perbatasan
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labeling
_, markers = cv2.connectedComponents(sure_fg)

# Tambahkan satu ke semua marker sehingga latar belakang menjadi 1
markers = markers + 1

# Tandai area perbatasan dengan 0
markers[unknown == 255] = 0

# Terapkan algoritma Watershed
markers = cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]  # Tandai perbatasan dengan warna merah

# Tampilkan hasil segmentasi
cv2.imshow('Watershed Results', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
