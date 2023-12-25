import tensorflow as tf
import time
#from tensorflow.python.client import device_lib
#from tensorflow.keras import layers, models

#print(device_lib.list_local_devices())

#sess = tf.Session(config=tf.ConfigProto(device_count={'DML': 1, 'GPU': 1, 'CPU': 1}))
#sess = tf.Session(config=tf.ConfigProto(device_count={'DML': 1, 'CPU': 1}))
#sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

# Генерируем простые данные для задачи классификации
(X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], -1)).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

# Простая модель для классификации
def create_simple_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Создаем модель
input_shape = X_train.shape[1:]
num_classes = 10
model = create_simple_model(input_shape, num_classes)

# Измеряем время обучения на CPU
start_time = time.time()
with tf.device('/CPU:0'):
    model.fit(X_train, y_train, epochs=5, batch_size=32)
end_time = time.time()
cpu_time = end_time - start_time
print(f"Training time on CPU: {cpu_time} seconds")

# Пересоздаем модель для чистоты эксперимента
model = create_simple_model(input_shape, num_classes)

# Измеряем время обучения на GPU
start_time = time.time()
with tf.device('/GPU:0'):
    model.fit(X_train, y_train, epochs=5, batch_size=32)
end_time = time.time()
gpu_time = end_time - start_time
print(f"Training time on GPU: {gpu_time} seconds")

# Пересоздаем модель для чистоты эксперимента
model = create_simple_model(input_shape, num_classes)

# Измеряем время обучения на DML
start_time = time.time()
with tf.device('/DML:0'):
    model.fit(X_train, y_train, epochs=5, batch_size=32)
end_time = time.time()
gpu_time = end_time - start_time
print(f"Training time on DML: {gpu_time} seconds")
