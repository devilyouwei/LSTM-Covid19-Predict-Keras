import csv
import numpy
import array

#数据转换为矩阵
def create_dataset(dataset, window=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-window-1):
        #窗口数据
        dataX.append(dataset[i:(i+window)])
        #目标数据
        dataY.append(dataset[i + window])

    return numpy.array(dataX), numpy.array(dataY)

# get worlds data
def get_data():
    with open('./data/time_series_covid_19_confirmed.csv') as f:
        reader = csv.reader(f)
        result = list(reader)
        arr = {}
        del result[0]
        for item in result:
            name = item[1]
            del item[0]
            del item[0]
            del item[0]
            del item[0]
            intArr = list(map(int,item))
            arr[name]=numpy.array(intArr)

        return arr

# get US data
def get_US():
    with open('./data/time_series_covid_19_confirmed_US.csv') as f:
        reader = csv.reader(f)
        result = list(reader)
        arr = {}
        del result[0]
        for item in result:
            name = item[6]
            del item[0]
            del item[0]
            del item[0]
            del item[0]
            del item[0]
            del item[0]
            del item[0]
            del item[0]
            del item[0]
            del item[0]
            del item[0]
            intArr = list(map(int,item))
            if name in arr:
                a = numpy.array(intArr)
                b = numpy.array(arr[name])
                arr[name] = a + b
            else:
                arr[name] = numpy.array(intArr)

        return arr

