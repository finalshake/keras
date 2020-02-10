import keras 
import numpy as np
import matplotlib.pyplot as plt
#顺序模型
from keras.models import Sequential
#全连接层
from keras.layers import Dense

#拟合的数据
x_data = np.random.rand(100)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = 2.1 * x_data + 0.1 + noise
#test的数据
 #x_test = np.random.rand(200)
 #test_noise = np.random.normal(0, 0.5, x_test.shape)
 #y_test = 0.6 * x_test + 10 + test_noise

#显示数据图形
plt.scatter(x_data, y_data)
plt.show()

#构建模型
model = Sequential()
#加一全连接层
model.add(Dense(units = 1, input_dim = 1))

#sgd: Stochastic gradient descent 随机梯度下降法
#mse: Mean Squared Error 平均误差
model.compile(optimizer = 'sgd', loss = 'mse')


# train the model
for step in range(4000):
    cost = model.train_on_batch(x_data, y_data)
    if step % 10 == 0:
        print('cost', cost)

W, b = model.layers[0].get_weights()
print('W:', W, 'b:', b)

y_pred = model.predict(x_data)

plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred, 'r-', lw=2)
plt.show()
