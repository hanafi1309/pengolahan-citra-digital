from email.mime import image
import cv2
import numpy as np

#membaca gambar
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_4/rock.jpg") 

#Mendefinisikan empat titik sudut citra asli
points1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 300]])

#mendefinisikan empat titik susdut baru
points2 = np.float32([[0, 0], [300,0], [0, 300] , [300, 300]])

#mendapatkan matriks transformasi perspektif
M_perspective = cv2.getPerspectiveTransform(points1, points2)

#melakukan tranformasi perspective
perspective_transform_image = cv2.warpPerspective(image, M_perspective, (300, 300))

#menammpilkan hasil 
cv2. imshow('perspective Transformed image', perspective_transform_image)
cv2.waitKey(0)
cv2.destroyAllWindows()