import cv2

#membaca gambar 
image = cv2.imread ("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_1/rock.jpg")

#ukuran baru untuk gambar 
new_width = 900
new_height = 1600

#mengubah ukuran gambar 
resized_image = cv2.resize(image, (new_width, new_height))

# menamppilkan gambar asli dan gambar yang telah di ubah ukuranya 
cv2.imshow ('Original Image ', image)
cv2.imshow ('Resized Image', resized_image)

#menunggu hingga ada input dari keyboard
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('resized_image.jpg', resized_image)
