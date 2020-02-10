import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

#构建测试数据集
x_data = np.linspace(-0.6, 0.6, 200)
noise = np.random.normal(0, 0.03, x_data.shape)
y_data = np.square(x_data) + noise

#画图
plt.scatter(x_data, y_data)
plt.show()

#构建模型
sgd = SGD(lr = 0.1)
model = keras.Sequential()
model.add(Dense(units = 10, activation = 'tanh', input_dim = 1))
model.add(Dense(units = 1, activation = 'tanh'))
model.compile(optimizer=sgd, loss='mse')


#训练模型
for step in range(6000):
    cost = model.train_on_batch(x_data, y_data)
    if step % 50 == 0:
        print('cost', cost)

# W, b = model.get_weights()
# print('W:', W, 'b:', b)

#测试模型
y_pred = model.predict(x_data)

plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred, 'r-', lw=2)
plt.show()
