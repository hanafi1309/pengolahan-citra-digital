import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def load_data(image_folder="images"):
    images = []
    labels = []
    
    if not os.path.exists(image_folder):
        raise ValueError(f"Folder '{image_folder}' tidak ditemukan!")

    for i in range(1, 11):  # Assuming 10 images: ruang_1.jpg, ..., ruang_10.jpg
        img_path = os.path.join(image_folder, f"ruang_{i}.jpg")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Gagal memuat gambar: {img_path}")
            continue
        
        images.append(img.flatten() / 255.0)
        labels.append(i)
    
    if not images:
        raise ValueError("Tidak ada gambar yang berhasil dimuat!")
    
    return np.array(images), np.array(labels)

# Load data
X, y = load_data("images")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Test model
accuracy = knn.score(X_test, y_test)
print(f'Akurasi KNN: {accuracy * 100:.2f}%')

