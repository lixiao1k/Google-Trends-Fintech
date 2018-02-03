import Census_Data as cd
import math
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

def loadNumberSold():
    tmp = np.loadtxt("numberSold.csv", dtype=np.str, delimiter=",")
    not_adjusted_num = tmp[13:73, 1].astype(np.float)
    adjusted_num = tmp[13:73, 2].astype(np.float)
    return not_adjusted_num, adjusted_num

def getRowData():
    a, b = loadNumberSold()
    return b


def getModelOneTrainData():
    # y(t-1)
    y_s = []
    # yt
    y_t = []
    a = getRowData()
    for i in a:
        i = math.log(i)
        y_s.append([i])
        y_t.append(i)
    return y_s[:-3], y_t[3:]

def modelOne():
    regr = linear_model.LinearRegression()
    y_s, y_t = getModelOneTrainData()
    regr.fit(y_s, y_t)
    coef, intercept = regr.coef_, regr.intercept_
    print coef, intercept
    # [ 1.0309615] -0.230115263479
    #step three
    # [ 1.12833709] -0.937993407675
    a, b = getModelOneTrainData()
    print regr.predict(a)
    print b
    c = getRowData()
    e = []
    for i in c:
        e.append(math.log(i))
    plt.scatter(e[:-3], b, color='blue')
    plt.plot(e[:-3], regr.predict(a), color='red', linewidth=4)
    plt.show()

if __name__ == '__main__':
    modelOne()