import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def loadPriceSold():
    tmp = np.loadtxt("priceSold.csv", dtype=np.str, delimiter=",")
    median_price = tmp[1:, 1].astype(np.float)
    average_price = tmp[1:, 2].astype(np.float)
    return median_price, average_price

def loadNumberSold():
    tmp = np.loadtxt("numberSold.csv", dtype=np.str, delimiter=",")
    not_adjusted_num = tmp[1:, 1].astype(np.float)
    adjusted_num = tmp[1:, 2].astype(np.float)
    return not_adjusted_num, adjusted_num

def drawFigure1():
    x = np.arange(0, 72, 1)
    y1, y2 = loadNumberSold()
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    l1, = ax1.plot(x, y1, 'b')
    ax1.set_ylim([35, 130])

    ax2 = ax1.twinx()  # this is the important function
    l2, = ax2.plot(x, y2, 'r')
    ax2.set_xlim([-2, 71])
    ax2.set_ylim([400, 1400])

    ax2.set_xticks([0, 11, 23, 35, 47, 59, 71])
    ax2.set_xticklabels(['2003', '2004', '2005', '2006', '2007', '2008'])

    plt.legend([l1, l2], ['Not Seasonally Adjusted', 'Seasonally adjusted'], loc=3, fontsize='x-small')
    plt.show()

def drawFigure2():
    x = np.arange(0, 72, 1)
    y1, y2 = loadPriceSold()
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    l1, = ax1.plot(x, y1, 'b')
    ax1.set_ylim([179000, 265000])

    ax2 = ax1.twinx()  # this is the important function
    l2, = ax2.plot(x, y2, 'r')
    ax2.set_xlim([-2, 71])
    ax2.set_ylim([230000, 330000])

    ax2.set_xticks([0, 11, 23, 35, 47, 59, 71])
    ax2.set_xticklabels(['2003', '2004', '2005', '2006', '2007', '2008'])

    plt.legend([l1, l2], ['Median Sales Price', 'Average Sales Price'], loc=4, fontsize='x-small')
    plt.show()

if __name__ == '__main__':
    loadNumberSold()