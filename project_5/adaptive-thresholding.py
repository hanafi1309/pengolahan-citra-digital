
import cv2

#membaca gambar dalam grayscale
image = cv2.imread ("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_5/rock.jpg", 0)

#menerapkan adaptive thershoding
adaptive_thers = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

#menampilkan hasil
cv2.imshow('Adaptive Thresholding', adaptive_thers)
cv2.waitKey(0)
cv2.destroyAllWindows()