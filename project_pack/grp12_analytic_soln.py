# imports
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import shape
from numpy.lib.npyio import genfromtxt, load


def plot_benchmark():
    from_data = np.genfromtxt("cs_p.txt", delimiter=',', skip_header=1)
    length = shape(from_data)

    t0 = from_data[0, 0]
    # period_meas = from_data[-1, 0] - from_data[0, 0]
    #tot_seconds = period_meas*365*24*3600
    t = np.linspace(t0, from_data[-1, 0], num=length[0])

    p_ana = np.zeros(length[0])
    a = 2*(10 ** (-1))
    b = 1*(10 ** (-1))
    P0 = from_data[0, 1]

    for i in range(len(t)):
        time = (t[i] - t0)  # *365*24*3600
        print(time)
        p_ana[i] = ((-a/b)*(1-(np.exp(-b*time))) + P0)

    f, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(from_data[:, 0], from_data[:, 1], 'r.', label='given data')
    ax.plot(t, p_ana, 'b', label="analytical")
    ax.set_ylabel("Pressure")
    ax.set_xlabel("Year")
    ax.legend()

    # plt.show()

    conc_data = np.genfromtxt("cs_cc.txt", delimiter=',', skip_header=1)
    size = shape(conc_data)

    t0 = conc_data[0, 0]
    # period_meas = from_data[-1, 0] - from_data[0, 0]
    #tot_seconds = period_meas*365*24*3600
    t = np.linspace(t0, conc_data[-1, 0], num=size[0])

    c_ana = np.zeros(size[0])
    a = 1*(10 ** (-3))
    b = 1*(10 ** (-1))
    P0 = from_data[0, 1]

    for i in range(len(t)):
        time = (t[i] - t0)  # *365*24*3600
        print(time)
        c_ana[i] =

    f, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(conc_data[:, 0], conc_data[:, 1], 'r.', label='given data')
    # ax.plot(t, p_ana, 'b', label="analytical")
    ax.set_ylabel("solution of conc.")
    ax.set_xlabel("year")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    plot_benchmark()
