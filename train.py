import myfunc as func
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import myfunc as func
import time

# get world's data
data = func.get_data()
# get US data
data_us = func.get_US()

scaler = MinMaxScaler(feature_range=(0, 100))

# 30 days a train
window = 30
# new model
model = Sequential()
model.add(LSTM(128, input_shape=(1, window)))
model.add(Dense(1))
# NJ test data
dataset_us = scaler.fit_transform(data_us['New Jersey'].reshape(-1,1))

# loop train
for k in data.keys():
    # world train data
    dataset = scaler.fit_transform(data[k].reshape(-1,1))

    print('Train: '+k)
    trainX, trainY = func.create_dataset(dataset, window)
    testX, testY = func.create_dataset(dataset_us, window)

    trainX = numpy.reshape(trainX, (len(trainX), 1, window))
    testX = numpy.reshape(testX, (len(testX), 1, window))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=100, verbose=2, callbacks=None)

    # predict
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # 处理预测值
    print('Test NJ')
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)

    #评估模型
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[0]))
    print('Test Score: %.2f RMSE' % (testScore))

    print(len(testY))
    print(len(testPredict))
    # plot baseline and predictions
    plt.plot(testY) # NJ plot
    plt.plot(testPredict) # NJ predict
    plt.savefig('./charts/t_{}.jpg'.format(k))
    plt.close()

model.save('./models/covid.h5')
