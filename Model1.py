import Census_Data as cd
import math
from sklearn import linear_model
import matplotlib.pyplot as plt

def getRowData():
    a, b = cd.loadNumberSold()
    return b[12:]


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
    return y_s[:-1], y_t[1:]

def modelOne():
    regr = linear_model.LinearRegression()
    y_s, y_t = getModelOneTrainData()
    regr.fit(y_s, y_t)
    coef, intercept = regr.coef_, regr.intercept_
    # print coef, intercept
    # [ 1.0309615] -0.230115263479
    a, b = getModelOneTrainData()
    print regr.predict(a)
    c = getRowData()
    e = []
    for i in c:
        e.append(math.log(i))
    plt.scatter(e[:-1], b, color='blue')
    plt.plot(e[:-1], regr.predict(a), color='red', linewidth=4)
    plt.show()

if __name__ == '__main__':
    modelOne()