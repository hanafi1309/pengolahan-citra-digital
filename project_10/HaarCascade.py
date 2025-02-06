import cv2

# Muat file haar cade untuk deteksi wajah 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 

# baca cita 
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_10/ruang.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Deteksi wajah s
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# gambar bouding box di seketitar wajah terdeteksi 
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0), 0)

# tampilkan hasil 
cv2.imshow('Face Detection', image )
cv2.waitKey(0)
cv2.destroyAllWindows() 