
import matplotlib.pyplot as plt
import numpy as np

def loadAgencies():
    tmp = np.loadtxt("Real_Estate_Agencies.csv", dtype=np.str, delimiter=",")
    agencies = tmp[3:, 1].astype(np.float)
    num_sales = tmp[3:, 2].astype(np.float)
    return agencies, num_sales

def loadRentals():
    tmp = np.loadtxt("Apartments_Residential_Rentals.csv", dtype=np.str, delimiter=",")
    rentals = tmp[3:, 1].astype(np.float)
    num_sales = tmp[3:, 2].astype(np.float)
    return rentals, num_sales

def draw():
    x = np.arange(0, 261, 1)
    y1, y2 = loadRentals()
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    l1, = ax1.plot(x, y1, 'b')
    ax1.set_ylim([20, 100])

    ax2 = ax1.twinx()  # this is the important function
    l2, = ax2.plot(x, y2, 'r')
    ax2.set_xlim([-2, 261])
    ax2.set_ylim([350, 1400])

    ax2.set_xticks([0, 52, 104, 157, 209])
    ax2.set_xticklabels(['2004', '2005', '2006', '2007', '2008'])

    plt.legend([l1, l2], ['Google Trends', 'New House Sale'], loc=3, fontsize='x-small')
    plt.show()

if __name__ == '__main__':
    draw()