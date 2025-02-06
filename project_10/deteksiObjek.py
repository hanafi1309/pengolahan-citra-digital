import cv2
import numpy as np

# Muat model YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

# Perbaikan: Tangani indeks layer secara dinamis
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Baca gambar
img = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_10/ruang.jpg")
if img is None:
    print("Gambar tidak ditemukan! Periksa jalur file gambar.")
    exit()

height, width, channels = img.shape

# Preprocessing YOLO
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Analisis hasil
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Gambar kotak
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Tampilkan hasil
cv2.imshow("Deteksi Objek", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
