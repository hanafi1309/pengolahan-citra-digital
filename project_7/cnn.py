from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Input
from tensorflow.keras.utils import to_categorical
import numpy as np

# Inisialisasi model CNN
model = Sequential()

# Menggunakan Input shape di awal model
model.add(Input(shape=(64, 64, 3)))  # Layer pertama dengan Input shape

# Tambahkan convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Tambahkan convolutional layer lainnya
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))

# Output layer
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Menyiapkan data (misalnya dataset acak untuk contoh)
X_train = np.random.rand(1000, 64, 64, 3)  # 1000 gambar RGB 64x64
y_train = np.random.randint(0, 10, 1000)  # 1000 label antara 0 dan 9
y_train = to_categorical(y_train, num_classes=10)  # One-hot encoding

# Latih model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Misalnya, Anda juga dapat mengevaluasi model setelah pelatihan dengan data uji
# model.evaluate(X_test, y_test)
