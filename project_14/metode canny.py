import cv2

# baca citra dalam grayscale
image = cv2. imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_14/punk.jpg", cv2.IMREAD_GRAYSCALE)

# terapkan deteksi tepi dengan Canny
edges = cv2.Canny(image, 100, 200)

#tampilkan hasil deteksi tepi
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()