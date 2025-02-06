import cv2

#baca citra digital format rgb 
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_12/punk.jpg")

#konversi citra dari rgb ke ycbcr
ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

#Extrak channel Y(luminance)
y_channel = ycbcr_image[:,:,0]

#tampilkan channel y
cv2.imshow('Y channel', y_channel)
cv2.waitKey(0)
cv2.destroyWindow()

