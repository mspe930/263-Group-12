# imports
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import shape
from numpy.lib.npyio import genfromtxt, load


def plot_benchmark(length):
    '''--------------
    This function creates plots of the analytical solutions for the provided data
    
    INPUT:
    length: an integer variable, which will determine the time step to analyse over
    
    OUTPUT:
    p_ana: an array of integer variables 
            for the analytical solution for pressure over the time period from the data
    c_ana: an array of integer variables 
            for the analytical solution for concentration over the time period from the data
    --------------'''
    
    press_data = np.genfromtxt("cs_p.txt", delimiter=',', skip_header=1)
    
    t0 = press_data[0, 0]
    t = np.linspace(t0, press_data[-1, 0], num=length)

    p_ana = np.zeros(length[0])
    a = 2*(10 ** (-1))
    b = 1*(10 ** (-1))
    P0 = press_data[0, 1]

    for i in range(len(t)):
        time = (t[i] - t0)  # *365*24*3600
        p_ana[i] = ((-a/b)*(1-(np.exp(-b*time))) + P0)

    '''f, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(press_data[:, 0], press_data[:, 1],
            'r.', label='given pressure data')
    ax.plot(t, p_ana, 'b', label="analytical")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_xlabel("Year")
    ax.legend()
    
    if plot:
        plt.show()
    else:
        plt.savefig('263_press_benchmark.png', dpi=300)'''

    conc_data = np.genfromtxt("cs_cc.txt", delimiter=',', skip_header=1)
    length = shape(conc_data)

    t0 = conc_data[0, 0]
    t = np.linspace(t0, conc_data[-1, 0], num=length)

    c_ana = np.zeros(length[0])

    M0 = 8.3 * 10 ** 3
    C0 = conc_data[0, 1]
    k = 10 / M0
    d = 0.5 * 10 ** (-1)
    L = (k * conc_data[0, 1] - k) / (k + d)

    for i in range(len(t)):
        time = (t[i] - t0)  # * 365 * 24  # *3600
        c_ana[i] = ((k + d*C0) / (k + d)) + (L / np.exp((k+d)*time))

    '''f, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(conc_data[:, 0], conc_data[:, 1],
            'r.', label='given concentration data')
    ax.plot(t, c_ana, 'b', label="analytical")
    ax.set_ylabel("CO2 Concentration [wt%]")
    ax.set_xlabel("Year")
    ax.legend()

    if plot:
        plt.show()
    else:
        plt.savefig('263_concentration_benchmark.png', dpi=300)'''
    
    return p_ana, c_ana


if __name__ == "__main__":
    plot_benchmark()
