import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage import data, exposure

# Contoh membaca citra dan ekstraksi fitur HOG
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_6/rock.jpg", cv2.IMREAD_GRAYSCALE)

# Ekstraksi fitur HOG
fd, hog_image = hog(image, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)

# Misalnya x adalah array fitur dan y adalah label kelas citra
X = np.random.rand(10, 64)  # Misalnya 10 sampel
y = np.random.randint(0, 2, 10)

# Split data untuk pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inisialisasi model SVM
clf = svm.SVC(kernel='linear', random_state=42)

# Latih model
clf.fit(X_train, y_train)

# Prediksi menggunakan data uji
y_pred = clf.predict(X_test)

# Hitung Akuransi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akuransi: {accuracy * 100:.2f}%')
