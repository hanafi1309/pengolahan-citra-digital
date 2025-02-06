from skimage.feature import graycomatrix, graycoprops
import cv2

# Membaca gambar dalam mode grayscale
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_6/rock.jpg", 0)

# Cek apakah gambar berhasil dibaca
if image is None:
    print("Gambar tidak ditemukan!")
else:
    # Menghitung GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Menghitung fitur tekstur dari GLCM
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    print(f'Contrast: {contrast}, Energy: {energy}')

