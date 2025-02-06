import cv2

#baca citra dalam format rgb
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_12/punk.jpg")

#konversi citra dari rgb ke HSV
hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

#tampilkan citra hasil digital
cv2.imshow('HSV Image', hsv_image)
cv2.waitKey(0)
cv2.destroyWindow()
