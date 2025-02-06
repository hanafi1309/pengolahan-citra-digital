import cv2
import numpy as np 
from sklearn import svm
from sklearn.model_selection import train_test_split



# membaca dataset citra 
def load_data():
    # ganti dengan jalur data set yang sesuai 
    images = [] # list untuk citra 
    labels = [] # list untuk label 
    for i in range (1, 11): # misalnya 10 citra 
        image = cv2.imread(f'image_{i}.jpg', cv2.IMREAD_GRAYSCALE)
        images.append(image.flatten()) #ubah citra menjadi vektor id
        labels.append(i) # minsalnya label sesuai dengan nomor citra 
    return np.array(images), np.array(labels)
    
# muat data dan bagu menjadi data pelatihan dan pengujian 
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# latih model SVM
model = svm.SVC(kernel='linear')
model.fit(X_train, y_test)

#uji model 
accuracy = model.score(X_test, y_test)
print(f'Akuransi SVM: {accuracy * 100:.2f}%')
