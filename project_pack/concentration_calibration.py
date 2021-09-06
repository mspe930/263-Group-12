import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from concentration_model import *
import pressure_calibration

def concentration_model(t,M0,a,b,c,d,P0):
    pars = np.array([M0,a,b,c,d,P0])
    ts,Cs = solve_concentration_ode(f=concentration_ode,t0=ts_data[0],t1=t[-1],dt=0.2,C0=Cs_data[0],pars=pars)
    f = interp1d(ts,Cs,kind='linear',bounds_error=False)
    Cs = f(t)
    return Cs

def main():
    global ts_data, Cs_data
    data = np.genfromtxt('cs_cc.txt',dtype=float,delimiter=', ',skip_header=1).T
    ts_data = data[0,:]
    Cs_data = data[1,:]

    a,b,c,P0 = pressure_calibration.main()

    pars0 = [5.e+03, a, b, c, 5.e-01, P0]
    pars_error = [10e8, .001*a, .001*b, .001*c, 1.e00, .001*P0]

    bounds = ([sum(x) for x in zip(pars0,[-1*e for e in pars_error])], [sum(x) for x in zip(pars0,pars_error)] )

    p,_ = curve_fit(concentration_model,ts_data,Cs_data,p0=pars0,bounds=bounds)

    print("\nAdditional parameter estimations for concentration model:")
    print(" "*5, end="")
    print("d = {:1.2e}".format(p[4]))
    print(" "*4, end="")
    print("M0 = {:1.2e}\n".format(p[0]))

if __name__ == "__main__":
    main()