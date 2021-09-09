from concentration_model import interpolate_injection
import numpy as np
from matplotlib import pyplot as plt
from pressure_calibration import *
from pressure_model import *
from pressure_predictions import *

def plot_pressure_posterior(pars):
    ''' Generates four what-if scenarios (x4, x2, x1 and x0.5 injection rates) using posterior samples, and plots these alongside 
        our fitted pressure model and measured pressure data, given a set of model parameters.

        Parameters
        ----------
        pars : array-like
            List of best fit parameters for our pressure model of the form (M0, a, b, c, d, P0).

        Returns
        -------
        None 
    '''
    # creates new plot
    f,ax = plt.subplots(1,1)

    # fetches and plots measured data 
    ts_data,Ps_data = fetch_pressure_data()
    ax.plot(ts_data,Ps_data,'ro',label='Measured data')

    # solves fitted model and plots it alongside data 
    ts,Ps = solve_pressure_ode(pressure_ode,ts_data[0],ts_data[-1],0.2,Ps_data[0],pars=pars)
    ax.plot(ts,Ps,'k-',label='Best fit model') 

    # sets a sensible covariance matrix for data 
    cov = [[1.e-8]*6]*6

    # sets a sensible standard deviation for data measurements 
    sigma = [.25]*len(ts_data)

    # generates 100 posterior samples of our parameters
    pars_uncert = np.random.multivariate_normal(pars,cov,100)
    
    # plots the effect of halving injection rate using posteriors
    for p in pars_uncert:
        half_injection(ax,p,ts[-1],Ps[-1],lw=0.4,alpha=0.2,label='_nolabel_')
    ax.plot([],[],'y-',label='Half injections')
    
    # plots the effect of keeping injection rates const. using posteriors
    for p in pars_uncert:
        no_changes_injection(ax,p,ts[-1],Ps[-1],lw=0.4,alpha=0.2,label='_nolabel_')
    ax.plot([],[],'b-',label='No change')

    # plots the effect of doubling injection rates using posteriors
    for p in pars_uncert:
        double_injection(ax,p,ts[-1],Ps[-1],lw=0.4,alpha=0.2,label='_nolabel_')
    ax.plot([],[],'g-',label='Double injections')

    # plots the effect of quadrupling injection rates using posteriors
    for p in pars_uncert:
        quadruple_injection(ax,p,ts[-1],Ps[-1],lw=0.4,alpha=0.2,label='_nolabel_')
    ax.plot([],[],'m-',label='Quadruple injections')

    ax.set_xlabel('Year [A.D.]') # sets x axis label
    ax.set_ylabel('Reservoir pressure [MPa]')

    # adds title and legend and shows plot
    ax.set_title('Ohaaki LP pressure model: scenario forecasts with uncertainty')
    ax.legend()
    plt.show()



def main():
    pars = calibrate_pressure_model([5.e+03, 2.5e-3,  3.e-01, 8.e-04, 5.e-01,  6.17e+00])
    plot_pressure_posterior(pars)

if __name__ == "__main__":
    main()