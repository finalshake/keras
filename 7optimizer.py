import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import Adam


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1)/255.0
x_test = x_test.reshape(x_test.shape[0], -1)/255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

adam = Adam(lr = 0.01)

model = Sequential(
    )
model.add(Dense(units = 100, input_dim = 28*28, bias_initializer='one', activation='tanh', kernel_regularizer=l2(0.0003)))
model.add(Dense(units = 100, activation='relu', kernel_regularizer=l2(0.003)))
model.add(Dense(units = 10, activation='softmax', kernel_regularizer=l2(0.0003)))
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=30, epochs=10)

loss, accuracy = model.evaluate(x_test, y_test)
train_loss, train_accuracy = model.evaluate(x_train, y_train)

print('loss:', loss)
print('accuracy:', accuracy)
print('train_loss:', train_loss)
print('train_accuracy:', train_accuracy)
