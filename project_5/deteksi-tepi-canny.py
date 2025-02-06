from email.mime import image
import cv2

# Membaca gamabar dalam grayscale 
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_5/rock.jpg",0)

#membaca deteksi tepi Canny 
edges = cv2.Canny(image, 100, 200)

#Menampilkan hasil 
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()