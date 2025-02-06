import cv2
import numpy as np

#membaca gambar 
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_5/rock.jpg")
Z = image.reshape((-1, 3))

#konfersi ke float
Z = np.float32(Z)

# kereteria kmeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Konversi kembali ke uint8 dan reshape
center = np.uint8(center)
res = center[label.flatten()]
segmented_image = res.reshape((image.shape))

# Menampilkan Hasil 
cv2.imshow('Kmeans Segmentation', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()