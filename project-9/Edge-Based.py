import cv2

# Baca citra grayscale
image_path = "c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_9/punk.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Gambar tidak dapat dibaca. Periksa path atau format file.")
else:
    # Deteksi tepi menggunakan Canny
    edge = cv2.Canny(image, 100, 200)

    # Tampilkan hasil
    cv2.imshow('Edges Detected', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
