import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar dalam format RGB
image_rgb = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_2/rock.jpg")  # Pastikan gambar berada di direktori yang benar
# Mengonversi gambar dari RGB ke Grayscale
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

# Menampilkan gambar asli dan gambar grayscale
cv2.imshow('Original Image (RGB)', image_rgb)
cv2.imshow('Grayscale Image', image_gray)
cv2.waitKey(0)

# Menampilkan histogram untuk gambar RGB (dengan tiga saluran warna)
colors = ('b', 'g', 'r')
plt.figure(figsize=(10, 5))
for i, col in enumerate(colors):
    hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.title('Histogram of Original RGB Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

# Menampilkan histogram untuk gambar Grayscale
plt.hist(image_gray.ravel(), 256, [0, 256], color='black')
plt.title('Histogram of Grayscale Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

# Menutup semua jendela OpenCV
cv2.destroyAllWindows()
