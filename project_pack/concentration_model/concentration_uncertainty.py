from urllib import parse
import numpy as np
from concentration_model import *
from concentration_predictions import *
from concentration_calibration import concentration_model
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def plot_predictions_uncert(pars):
    # creates new plot
    f,ax = plt.subplots(1,1)

    # fetches and plots measured data 
    ts_data,Cs_data = fetch_concentration_data()
    ax.plot(ts_data,Cs_data,'ro',label='Measured data')

    # solves fitted model and plots it alongside data 
    ts,Cs = solve_concentration_ode(concentration_ode,ts_data[0],ts_data[-1],0.2,Cs_data[0],pars=pars)
    ax.plot(ts,Cs,'k-',label='Best fit model') 

    _,Ps_data = fetch_pressure_data()

    cov = [[1.e-4]*6]*6

    sigma = [.25]*len(ts_data)
    
    pars_uncert = np.random.multivariate_normal(pars,cov,100)
    
    for p in pars_uncert:
        half_injections(ax,p,ts[-1],Cs[-1],lw=0.4,alpha=0.2,label='_nolabel_')
    ax.plot([],[],'y-',label='Half injections')
    
    for p in pars_uncert:
        no_changes_injection(ax,p,ts[-1],Cs[-1],lw=0.4,alpha=0.2,label='_nolabel_')
    ax.plot([],[],'b-',label='No change')

    for p in pars_uncert:
        double_injection(ax,p,ts[-1],Cs[-1],lw=0.4,alpha=0.2,label='_nolabel_')
    ax.plot([],[],'g-',label='Double injections')

    for p in pars_uncert:
        quadruple_injection(ax,p,ts[-1],Cs[-1],lw=0.4,alpha=0.2,label='_nolabel_')
    ax.plot([],[],'m-',label='Quadruple injections')
    

    ax.legend()
    plt.show()

def main():
    pars0 = [5.88e+03, 1.92e-03,  1.41e-01, 8.80e-04, 4.11e-01,  6.17e+00]
    plot_predictions_uncert(pars0)

if __name__ == "__main__":
    main()
