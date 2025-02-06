import cv2

# Membaca gambar
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_6/rock.jpg")

# Cek apakah gambar berhasil dibaca
if image is None:
    print("Error: Gambar tidak ditemukan atau tidak bisa dibaca.")
    exit()

# Inisialisasi objek SURF
surf = cv2.xfeatures2d.SURF_create()

# Mendeteksi keypoints dan deskriptor
keypoints, descriptors = surf.detectAndCompute(image, None)

# Menggambar keypoints di citra
surf_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Menampilkan hasil
cv2.imshow('SURF Features', surf_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
