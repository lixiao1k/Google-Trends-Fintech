import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import math

def getAveragePrice():
    tmp = np.loadtxt("priceSold.csv", dtype=np.str, delimiter=",")
    average_price = tmp[1:, 2].astype(np.float)[12:-1]

    result = []
    for i in average_price:
        result.append(math.log(i))
    return result

def getOtherData():
    tmp = np.loadtxt("numberSold.csv", dtype=np.str, delimiter=",")
    adjusted_num = tmp[13:, 2].astype(np.float)
    rentals = tmp[13:, 3].astype(np.float)
    agencies = tmp[13:, 4].astype(np.float)
    return adjusted_num, rentals, agencies

def getAdjusted_lastNum():
    a, b, c = getOtherData()
    adjusted_num = []
    for i in a:
        adjusted_num.append(math.log(i))
    return adjusted_num[:-3]

def getAdjusted_nowNum():
    a, b, c = getOtherData()
    adjusted_num = []
    for i in a:
        adjusted_num.append(math.log(i))
    return adjusted_num[3:]

def getRentals_now():
    a, rentals, c = getOtherData()
    return rentals[3:]

def getAgency_now():
    a, b, agencies = getOtherData()
    return agencies[3:]

def getXdata():
    data = []
    adjustedlast_num = getAdjusted_lastNum()
    rentals_now = getRentals_now()
    agencies_now = getAgency_now()
    average_price = getAveragePrice()

    for i in range(len(adjustedlast_num)):
        tmp = []
        tmp.append(adjustedlast_num[i])
        tmp.append(rentals_now[i])
        tmp.append(agencies_now[i])
        # tmp.append(average_price[i])
        data.append(tmp)
    return data


def returnToOrignal(data):
    result = []
    for i in data:
        result.append(math.exp(i))
    return result



def model2():
    regr = linear_model.LinearRegression()
    regr.fit(getXdata(), getAdjusted_nowNum())
    coef, intercept = regr.coef_, regr.intercept_
    print coef, intercept
    # [1.03196516e+00   6.74296277e-05 - 2.89076345e-04 - 6.58578381e-07] - 0.0320975290802
    # if use average price of last month
    # [1.03199334e+00  -1.18092902e-04  -1.53651831e-05  -5.19698019e-07] -0.0749491132377
    # if drop average price
    # [  1.02678202e+00  -4.16507922e-04   4.36685689e-04] -0.195598719278
    # step three
    # [ 1.11925751 -0.00237715  0.00204901] -0.81473550336
    x = np.arange(0, 57, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, returnToOrignal(getAdjusted_nowNum()), linewidth=2)
    plt.plot(x, returnToOrignal(regr.predict(getXdata())), 'r-o', linewidth=2)
    print regr.predict(getXdata())
    ax.set_xticks([0, 11, 23, 35, 47])
    ax.set_xticklabels(['2004', '2005', '2006', '2007', '2008'])
    ax.set_xlim([-2, 59])
    ax.set_ylim([350, 1400])
    plt.ylabel("Seasonally Adjusted Annual Sales Rate in 1,000")
    plt.show()


if __name__ == '__main__':
    model2()
    # print math.log(226000)
    # print math.log(1165)*1.03196516e+00 + 80*6.74296277e-05 - 70*2.89076345e-04 - 262100*6.58578381e-07 - 0.0320975290802
    # print getAdjusted_nowNum()
    # print getAveragePrice()