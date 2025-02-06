import cv2

# Membaca gambar
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_6/rock.jpg", cv2.IMREAD_GRAYSCALE)

# Membuat objek SIFT
sift = cv2.SIFT_create()

# Deteksi keypoints dan deskriptor
keypoints, descriptors = sift.detectAndCompute(image, None)

# Menampilkan keypoints pada gambar
output_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Menampilkan hasil
cv2.imshow("SIFT Keypoints", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
