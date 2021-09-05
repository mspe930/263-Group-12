import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pressure_model import *

def pressure_model(t,M0,a,b,c,d,P0):
    pars = np.array([M0,a,b,c,d,P0])
    ts,Ps = solve_pressure_ode(f=pressure_ode,t0=ts_data[0],t1=t[-1],dt=0.2,P0=Ps_data[0],pars=pars)
    f = interp1d(ts,Ps,kind='linear',bounds_error=False)
    Ps = f(t)
    return Ps

def main():
    # define global variables for measured data
    global ts_data, Ps_data
    # read pressure data from file
    data = np.genfromtxt('cs_p.txt',dtype=float,delimiter=', ',skip_header=1).T
    # store lists of data as global variables
    ts_data = data[0,:]
    Ps_data = data[1,:]

    # INITIAL PARAMETER GUESS of the form (M0, a, b, c, d, P0)
    pars0 = [5.00000000e+03,  4.97642037e-03,  4.99956037e-01, -1.61422003e-03,
  5.00000000e-01,  6.16997956e+00]

    # set bounds for parameter estimation
    # P0 is fixed for now so has a very small bounding radius +/- 0.01 MPa
    bounds = ([-1e8]*5 + [Ps_data[0]-0.01], [1e8]*5 + [Ps_data[0]+0.01])

    # using gradient descent method to compute the best fit parameters
    p,_ = curve_fit(pressure_model,ts_data,Ps_data,p0=pars0,bounds=bounds)

    print("Best fit parameter estimation for pressure model: ")
    print(" "*5, end="")
    print("a = {:1.2e}".format(p[1]))
    print(" "*5, end="")
    print("b = {:1.2e}".format(p[2]))
    print(" "*5, end="")
    print("c = {:1.2e}".format(p[3]))


if __name__ == "__main__":
    main()