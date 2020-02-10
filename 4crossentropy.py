import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)/255.0
x_test = x_test.reshape(x_test.shape[0], -1)/255.0

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


sgd = SGD(lr=0.2)
model = Sequential()
model.add(Dense(units=10, activation = 'softmax', bias_initializer = 'one', input_dim = 28*28))
#loss 改成交叉商
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=30, epochs=10)

loss, accuracy = model.evaluate(x_test, y_test)

print('loss:', loss)
print('accuracy:', accuracy)
