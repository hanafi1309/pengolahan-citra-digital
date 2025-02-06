import cv2
import numpy as np

# Contoh pemrosesan citra
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_6/rock.jpg")  # Ganti dengan path citra Anda
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Corner detection (contoh menggunakan Harris)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Buat mask dari deteksi
mask = dst > 0.01 * dst.max()

# Tetapkan nilai merah (BGR) pada lokasi yang memenuhi mask
image[mask] = [0, 0, 255]

# Tampilkan atau simpan hasil
cv2.imshow('Detected Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


