from tensorflow.keras import layers, models
import tensorflow as tf

# Menggunakan dataset CIFAR-10 sebagai contoh
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalisasi data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Membangun model data
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Melatih model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
