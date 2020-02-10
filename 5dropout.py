import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#导入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#打印数据格式
print('x_shape:', x_train.shape)
print('y_shape', y_train.shape)
# print(x_train[0])
# print(y_train)

#reshape the data
x_train = x_train.reshape(x_train.shape[0], -1)/255.0
x_test = x_test.reshape(x_test.shape[0], -1)/255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
# print(y_train)


#构建模型
sgd = SGD(lr=0.1)
model = Sequential([Dense(units=100, input_dim=28*28, bias_initializer='one', activation='tanh')])
model.add(Dropout(0.3))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

#训练模型
model.fit(x_train, y_train, batch_size=30, epochs=10)

#评估模型
loss, accuracy = model.evaluate(x_test, y_test)
train_loss, train_accuracy = model.evaluate(x_train, y_train)
print('test loss:', loss)
print('accuracy:', accuracy)
print('train loss:', train_loss)
print('train accuracy:', train_accuracy)
