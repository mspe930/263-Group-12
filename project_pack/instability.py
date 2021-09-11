from re import L
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import shape
from numpy.lib.npyio import genfromtxt, load
from scipy.interpolate import interp1d
import pressure_calibration
import concentration_calibration
import analytic_soln
from pressure_model import *
from concentration_model import *


def pressure_time_step(t, M0, a, b, c, d, P0, t_step):
    '''function solves for the numeric solution of pressure at a variable time step
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
    t_step : int
        Determines the step size used when solving the ODE
    Returns:
    ---------
    Ps : array-like
        List of pressures for given parameters and paired times.
    '''

    # store input parameters as an array
    pars = np.array([M0, a, b, c, d, P0])
    # read pressure data
    ts_data, Ps_data = fetch_pressure_data()
    # use pressure model to compute pressure at a range of times
    ts, Ps = solve_pressure_ode(
        f=pressure_ode, t0=ts_data[0], t1=t[-1], dt=t_step, P0=Ps_data[0], pars=pars)
    # interpolate solved pressures to the list of given time data
    f = interp1d(ts, Ps, kind='linear', bounds_error=False)
    Ps = f(t)

    # return list of pressures at the given list of time data
    return Ps


def concentration_time_step(t, M0, a, b, c, d, P0, t_step):
    '''function solves for the numeric solution of concentration at a large time step
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
    t_step : int
        Determines the step size used when solving the ODE
    Returns:
    ---------
    Cs : array-like
        List of pressures for given parameters and paired times.
    '''

    # store input parameters as an array
    pars = np.array([M0, a, b, c, d, P0])
    # read pressure data
    ts_data, Cs_data = fetch_concentration_data()
    # use concentration model to compute pressure at a range of times
    ts, Cs = solve_concentration_ode(
        f=concentration_ode, t0=ts_data[0], t1=t[-1], dt=t_step, C0=Cs_data[0], pars=pars)
    # interpolate solved concentrations to the list of given time data
    f = interp1d(ts, Cs, kind='linear', bounds_error=False)
    Cs = f(t)

    # return list of concentrations at the given list of time data
    return Cs


def plot_pressure_time_step(M0, a, b, c, d, P0, t_step):
    '''function plots the numeric, analytical, and steady state solutions of pressure ODE
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
    t_step : int
        Determines the step size used when solving the ODE
    Returns:
    ---------
    None
    '''

    # store input parameters as an array
    pars = np.array([M0, a, b, c, d, P0])
    # read pressure data
    t, p = fetch_pressure_data()
    # solve for analytical solution
    t_ana, p_ana = analytic_soln.solve_pressure_benchmark(*pars)
    # set steady state to be initial pressure value
    p_steady = np.full((np.size(t)), p[0])

    # solve for numerical solution using larger time step
    p_num = pressure_time_step(t, *pars, t_step)

    # plot the analytical, numerical and steady state solutions
    f, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(t[:], p_num, 'kx', label="numerical solution")
    ax.plot(t_ana, p_ana, 'b', label="analytical solution")
    ax.plot(t, p_steady, 'b--', label="steady state")
    ax.set_title("Pressure ODE: instability at large time-step")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_xlabel("Time [Year]")
    ax.legend()

    plt.show()


def plot_concentration_time_step(M0, a, b, c, d, P0, t_step):
    '''function plots the numeric, analytical, and steady state solutions of concentration ODE
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
    t_step : int
        Determines the step size used when solving the ODE
    Returns:
    ---------
    None
    '''

    # store input parameters as an array
    pars = np.array([M0, a, b, c, d, P0])
    # read concentration data
    t, c = fetch_concentration_data()
    # solve for analytical solution
    t_ana, c_ana = analytic_soln.solve_concentration_benchmark(*pars)
    # set steady state as initial measurement
    c_steady = np.full((np.size(t)), c[0])

    # solve for numerical with variable time step
    c_num = concentration_time_step(t, *pars, t_step)

    # plot the solutions
    f, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(t[:], c_num, 'kx', label="numerical solution")
    ax.plot(t_ana, c_ana, 'b', label="analytical solution")
    ax.plot(t, c_steady, 'b--', label="steady state")
    ax.set_title("Concentration ODE: instability at large time-step")
    ax.set_ylabel("CO2 concentration [wt%]")
    ax.set_xlabel("Time [Year]")
    ax.legend()

    plt.show()


def main():
    # inital parameter guess
    pars0 = [5.e+03, 2.5e-3,  3.e-01, 8.e-04, 5.e-01,  6.17e+00]
    # plot pressure at large time step,
    plot_pressure_time_step(*pars0, 10)
    # plot concentration at large time step
    plot_concentration_time_step(*pars0, 10)


if __name__ == "__main__":
    main()