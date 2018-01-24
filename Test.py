import numpy as np
import math
from Model2 import getOtherData
import matplotlib.pyplot as plt

def getSlodNum_Last():
    tmp = np.loadtxt("numberSold_test.csv", dtype=np.str, delimiter=",")
    soldNum_last = tmp[1:-3, 2].astype(np.float)
    return soldNum_last

def getSoldNum_Now():
    tmp = np.loadtxt("numberSold_test.csv", dtype=np.str, delimiter=",")
    soldNum_now = tmp[4:, 2].astype(np.float)
    return soldNum_now

def getRentals_Now():
    tmp = np.loadtxt("numberSold_test.csv", dtype=np.str, delimiter=",")
    rentals_now = tmp[4:, 3].astype(np.float)
    return rentals_now

def getAgency_Now():
    tmp = np.loadtxt("numberSold_test.csv", dtype=np.str, delimiter=",")
    agency_now = tmp[4:, 4].astype(np.float)
    return agency_now


def predict():
    predict_result = []
    soldNum_last = getSlodNum_Last()
    soldNum_now = getSoldNum_Now()
    rentals_now = getRentals_Now()
    agency_now = getAgency_Now()
    for i in range(len(soldNum_last)):
        # tmp = math.log(soldNum_last[i]) * 1.02678202e+00 - 4.16507922e-04 * rentals_now[i] + 4.36685689e-04 * agency_now[i] - 0.195598719278
        # step three
        tmp = math.log(soldNum_last[i]) * 1.11925751 - 0.00237715 * rentals_now[i] + 0.00204901 * agency_now[i] - 0.81473550336
        predict_result.append(math.exp(tmp))
    return predict_result, soldNum_now

def getAAPE():
    predict_result, soldNum_real = predict()
    tmp = 0.0
    for i in range(len(predict_result)):
        tmp = tmp + abs((predict_result[i]-soldNum_real[i])/soldNum_real[i])
    # 7.16258275472
    # Step three
    # 15.8459857534
    return tmp/len(predict_result) * 100

def draw():
    a, b, c = getOtherData()
    train_data = []
    for i in a:
        train_data.append(i)
    real_now = getSoldNum_Now()
    for i in real_now:
        train_data.append(i)
    predict_data = []
    for i in a:
        predict_data.append(i)
    predict_result, soldNum_real = predict()
    for i in predict_result:
        predict_data.append(i)

    x = np.arange(0, 132, 1)
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    l1, = ax1.plot(x, train_data, 'b', linewidth=2)
    plt.ylabel("Number of House sold")

    y = np.arange(60, 132, 1)
    ax2 = ax1.twinx()  # this is the important function
    l2, = ax2.plot(y, predict_result, 'r', linewidth=2)
    ax2.set_ylim([200, 1400])

    plt.legend([l1, l2], ['Real Data', 'Predicted Data'], loc=1, fontsize='x-small')
    plt.plot([60, 60], [0, 1400], color ='black', linewidth=2.5, linestyle="--")
    plt.annotate('left is training data',
             xy=(60, 1000), xycoords='data',
             xytext=(+10, +30), textcoords='offset points', fontsize=12,
             arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    ax1.set_xticks([0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120])
    ax1.set_xticklabels(['2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014'])
    ax1.set_xlim([-3, 132])
    plt.show()

if __name__ == '__main__':
    print getAAPE()
    # draw()