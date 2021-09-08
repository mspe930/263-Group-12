import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from concentration_model import *
from pressure_calibration import calibrate_pressure_model

def concentration_model(t,M0,a,b,c,d,P0):

    pars = np.array([M0,a,b,c,d,P0])
    ts_data,Cs_data = fetch_concentration_data()
    ts,Cs = solve_concentration_ode(f=concentration_ode,t0=ts_data[0],t1=t[-1],dt=0.2,C0=Cs_data[0],pars=pars)
    f = interp1d(ts,Cs,kind='linear',bounds_error=False)
    Cs = f(t)
    return Cs

def compute_residuals(Cs):
    ts_data,Cs_data = fetch_concentration_data()
    residuals = np.zeros(len(ts_data))
    for i in range(len(ts_data)):
        residuals[i] = Cs[i] - Cs_data[i]
    return residuals

def plot_concentration_residuals(pars):
    ts_data,Cs_data = fetch_concentration_data()
    ts,Ps = solve_concentration_ode(f=concentration_ode,t0=ts_data[0],t1=ts_data[-1],dt=0.2,C0=Cs_data[0],pars=pars)
    f = interp1d(ts,Ps,kind='linear',bounds_error=False)
    Cs = f(ts_data)
    residuals = compute_residuals(Cs)

    f,ax = plt.subplots(1,1)
    ax.plot(ts_data,residuals,'ro',markerSize=8,markerfacecolor='none',label='Fitted Model:\n M0 = {:1.2e}\n a = {:1.2e}\n b = {:1.2e}\n c = {:1.2e}\n d = {:1.2e}'.format(*pars[:-1]))
    ax.set_xlabel('Year of observation [A.D.]')
    ax.set_ylabel('Residuals [MPa]')
    ax.set_title('Residuals plot of fitted LP CO2 concentration model')
    ax.legend()
    plt.show()


def calibrate_concentration_model(pars0):
    ts_data,Cs_data = fetch_concentration_data()
    p = calibrate_pressure_model(pars0)
    lwr_bnds = [0., .999*p[1], .999*p[2], .999*p[3], 0., .999*p[5]]
    upr_bnds = [10.e10, 1.001*p[1], 1.001*p[2], 1.001*p[3], 10.e10, 1.001*p[5]]
    bounds = (lwr_bnds, upr_bnds)

    p,_ = curve_fit(concentration_model,ts_data,Cs_data,p0=p,bounds=bounds)
    print("\nAdditional parameter estimations for concentration model:")
    print(" "*5, end="")
    print("d = {:1.2e}".format(p[4]))
    print(" "*4, end="")
    print("M0 = {:1.2e}\n".format(p[0]))

    return p

def main():
    pars0 = [8.e+03, 2.5e-3,  3.e-01, 8.e-04, 5.e-01,  6.17e+00]
    pars = calibrate_concentration_model(pars0)
    plot_concentration(pars)
    plot_concentration_residuals(pars)

if __name__ == "__main__":
    main()