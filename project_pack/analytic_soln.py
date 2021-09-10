# imports
from re import L
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import shape
from numpy.lib.npyio import genfromtxt, load
import pressure_calibration
import concentration_calibration


def solve_pressure_benchmark(M0, a, b, c, d, P0):
    '''--------------
    This function solves the analytical solution for the pressure ODE

    Parameters
    ----------
    t : array-like
        List of time data to compute the reservoir pressure at.
    M0 : float
        Parameter - initial mass of CO2 in reservoir. This is unused in the pressure model but is required for
        consistency with our concentration model.
    a : float
        Source/sink strength lumped parameter.
    b : float
        Recharge strength lumped parameter.
    c : float
        Slow drainage strength lumped parameter.
    d : float
        Diffusion strength lumped parameter. Unused in pressure model but required for consistency with our concentration model.
    P0 : float
        Parameter - initial reservoir pressure.

    Returns
    ----------
    t : array-like
        values of time pressure is solved at
    p_ana : array-like
        solved analytical values of pressure
    '''

    # retrieve pressure data from supplied file, save data into 2D array
    press_data = np.genfromtxt("cs_p.txt", delimiter=',', skip_header=1)

    # set t0 to be inital value from the given data
    t0 = press_data[0, 0]
    # create t, which is time values to solve at, over range of given data
    t = np.linspace(t0, press_data[-1, 0], 80)

    # initialise the array for solved pressure values
    p_ana = np.zeros(80)

    # initialise constants
    a = 2.5*(10 ** (-1))
    b = 1 * (10 ** (-1))

    # set initial pressure equal to one from given data
    P0 = press_data[0, 1]

    # for every value of time (which is since the initial time) solve for p using the simplified benchmarking equation
    for i in range(len(t)):
        time = (t[i] - t0)  # *365*24*3600
        p_ana[i] = ((-a/b)*(1-(np.exp(-b*time))) + P0)

    return(t, p_ana)


def solve_concentration_benchmark(M0, a, b, c, d, P0):
    '''--------------
    This function solves the analytical solution for the concentration ODE

    Parameters
    ----------
    t : array-like
        List of time data to compute the reservoir pressure at.
    M0 : float
        Parameter - initial mass of CO2 in reservoir. This is unused in the pressure model but is required for
        consistency with our concentration model.
    a : float
        Source/sink strength lumped parameter.
    b : float
        Recharge strength lumped parameter.
    c : float
        Slow drainage strength lumped parameter.
    d : float
        Diffusion strength lumped parameter. Unused in pressure model but required for consistency with our concentration model.
    P0 : float
        Parameter - initial reservoir pressure.

    Returns
    ---------
    t : array-like
        values of time concentration is solved at
    c_ana : array-like
        solved analytical values of concentration

    '''
    # retrieve concentration data from supplied data file
    conc_data = np.genfromtxt("cs_cc.txt", delimiter=',', skip_header=1)

    # initialise time as being beginning of supplied data
    t0 = conc_data[0, 0]
    # initialise time array over range of given data
    t = np.linspace(t0, conc_data[-1, 0], 80)

    # initialise array of analytic solution
    c_ana = np.zeros(80)

    # initialise constants
    C0 = conc_data[0, 1]
    k = 7/M0
    d = 0.5 * 10 ** (-1)
    L = (k * C0 - k) / (k + d)

    # for each time value solve for concentration value using benchmarking equation
    for i in range(len(t)):
        # use time as time since beginning of data
        time = (t[i] - t0)  # * 365 * 24  # *3600
        c_ana[i] = ((k + d*C0) / (k + d)) + (L / np.exp((k+d)*time))

    return(t, c_ana)


def plot_pressure_benchmark(M0, a, b, c, d, P0):
    '''--------------
    This function creates plots of the analytical,numerical and steady state solution for the provided concentration data

    Parameters
    ----------
    M0 : float
        Parameter - initial mass of CO2 in reservoir. This is unused in the pressure model but is required for
        consistency with our concentration model.
    a : float
        Source/sink strength lumped parameter.
    b : float
        Recharge strength lumped parameter.
    c : float
        Slow drainage strength lumped parameter.
    d : float
        Diffusion strength lumped parameter. Unused in pressure model but required for consistency with our concentration model.
    P0 : float
        Parameter - initial reservoir pressure.

    Returns
    ---------
    None
    '''

    # store parameters in array
    pars = [M0, a, b, c, d, P0]

    # retrieve pressure data from supplied file, save data into 2D array
    press_data = np.genfromtxt("cs_p.txt", delimiter=',', skip_header=1)
    # retrieve numerical solution from function 'pressure_calibration'
    p_num = pressure_calibration.pressure_model(
        press_data[:, 0], M0, a, b, c, d, P0)

    # steady state solution as initial pressure value
    p_steady = np.full((80), P0)

    # find solutions of benchmarked ODE
    t, p_ana = solve_pressure_benchmark(*pars)

    # plot the numerical, analytical, and steady state solutions on the same axes
    # all labelled respectively
    f, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(t, p_ana, 'b', label="analytical solution")
    ax.plot(press_data[:, 0], p_num, 'k-', label="numerical solution")
    ax.plot(t, p_steady, 'b--', label="steady state")
    ax.set_title("benchmark: a=0.25, b=0.1, c=0.00")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_xlabel("Year")
    ax.legend()

    plt.show()


def plot_concentration_benchmark(M0, a, b, c, d, P0):
    '''--------------
    This function creates plots of the analytical, numerical and steady state solution for the provided pressure data

    Parameters
    ----------
    M0 : float
        Parameter - initial mass of CO2 in reservoir. This is unused in the pressure model but is required for
        consistency with our concentration model.
    a : float
        Source/sink strength lumped parameter.
    b : float
        Recharge strength lumped parameter.
    c : float
        Slow drainage strength lumped parameter.
    d : float
        Diffusion strength lumped parameter. Unused in pressure model but required for consistency with our concentration model.
    P0 : float
        Parameter - initial reservoir pressure.

    Returns
    ---------
    None
    '''

    # store parameters in array
    pars = [M0, a, b, c, d, P0]

    # retrieve concentration data from supplied data file
    conc_data = np.genfromtxt("cs_cc.txt", delimiter=',', skip_header=1)
    # solve for numerical concentration, using function concentration_model from the file 'concentration_calibration'
    c_num = concentration_calibration.concentration_model(
        conc_data[:, 0], M0, a, b, c, d, P0)

    # create array of steady state solution from initial value of concentration
    c_steady = np.full((80), conc_data[0, 1])

    # find solution of benchmarked ODE
    t, c_ana = solve_concentration_benchmark(*pars)

    # plot analytic, numeric and steady state solutions
    f, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(t, c_ana, 'b', label="analytical solution")
    ax.plot(conc_data[:, 0], c_num, 'k-', label="numerical solution")
    ax.plot(t, c_steady, 'b--', label="steady state")
    ax.set_title("Benchmark: q(co2) = 7, q(water) = 0, d = 0.05")
    ax.set_ylabel("CO2 Concentration [wt%]")
    ax.set_xlabel("Year")
    ax.legend()

    plt.show()


def main():
    # initial guess of parameters
    pars0 = [5.e+03, 2.5e-3,  3.e-01, 8.e-04, 5.e-01,  6.17e+00]
    # plot pressure
    plot_pressure_benchmark(*pars0)
    # plot concentration
    plot_concentration_benchmark(*pars0)


if __name__ == "__main__":
    main()
