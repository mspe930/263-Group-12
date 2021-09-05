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
    global ts_data, Ps_data
    data = np.genfromtxt('cs_p.txt',dtype=float,delimiter=', ',skip_header=1).T
    ts_data = data[0,:]
    Ps_data = data[1,:]

if __name__ == "__main__":
    main()