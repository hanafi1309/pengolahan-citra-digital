import cv2

#baca  citra grayscale
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_11/punk.jpg", cv2.IMREAD_GRAYSCALE)

#Deteksi menggunakn Canny
edges = cv2.Canny(image, 100, 200)

#Tampilkan Hasil 
cv2.imshow('Canny Edge Detection ', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
