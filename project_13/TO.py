import cv2

#baca citra dalam grayscale
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_11/punk.jpg", cv2.IMREAD_GRAYSCALE)

#terapkan Otsu's threshold
ret, otsu_thresh = cv2.threshold(image,0, 255, cv2.THRESH_BINARY + cv2. THRESH_OTSU)

#tampilkan hasil segmentasi 
cv2.imshow('Otsu Thresholding', otsu_thresh)
cv2.waitKey(0)
cv2.destroyWindow()