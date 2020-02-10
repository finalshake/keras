import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

#导入数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#转变格式
x_train = x_train.reshape(-1, 28, 28, 1)/255.0
x_test = x_test.reshape(-1, 28, 28, 1)/255.0

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

#构建模型
model = Sequential()
model.add(Convolution2D(input_shape=(28, 28, 1), filters = 32, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2, strides = 2, padding = 'same'))
model.add(Convolution2D(filters  = 64, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#训练模型
adam = Adam(lr=0.1)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=10)

loss, accuracy = model.evaluate(x_test, y_test)

print('loss:', loss)
print('accuracy:', accuracy)
