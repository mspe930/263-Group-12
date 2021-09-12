from urllib import parse
import numpy as np
from concentration_model import *
from concentration_predictions import *
from concentration_calibration import concentration_model
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def plot_predictions_uncert(pars):
    ''' Generates four what-if scenarios (x4, x2, x1 and x0.5 injection rates) using posterior samples, and plots these alongside 
        our fitted concentration model and measured concentration data, given a set of model parameters.

        Parameters
        ----------
        pars : array-like
            List of best fit parameters for our concentration model of the form (M0, a, b, c, d, P0).

        Returns
        -------
        None
    '''
    # creates new plot
    f,ax = plt.subplots(1,1)

    # fetches and plots measured data 
    ts_data,Cs_data = fetch_concentration_data()
    ax.plot(ts_data,Cs_data,'ro',label='Measured data')

    # solves fitted model and plots it alongside data 
    ts,Cs = solve_concentration_ode(concentration_ode,ts_data[0],ts_data[-1],0.2,Cs_data[0],pars=pars)
    ax.plot(ts,Cs,'k-',label='Best fit model') 

    # sets a sensible covariance matrix for data 
    cov = [[1.e-4]*6]*6

    # sets a sensible standard deviation for data measurements 
    sigma = [.25]*len(ts_data)

    # generates 100 posterior samples of our parameters
    pars_uncert = np.random.multivariate_normal(pars,cov,100)
    
    # plots the effect of halving injection rate using posteriors
    for p in pars_uncert:
        half_injections(ax,p,ts[-1],Cs[-1],lw=0.4,alpha=0.2,label='_nolabel_')
    ax.plot([],[],'y-',label='Half injections')
    
    # plots the effect of keeping injection rates const. using posteriors
    for p in pars_uncert:
        no_changes_injection(ax,p,ts[-1],Cs[-1],lw=0.4,alpha=0.2,label='_nolabel_')
    ax.plot([],[],'b-',label='No change')

    # plots the effect of doubling injection rates using posteriors
    for p in pars_uncert:
        double_injection(ax,p,ts[-1],Cs[-1],lw=0.4,alpha=0.2,label='_nolabel_')
    ax.plot([],[],'g-',label='Double injections')

    # plots the effect of quadrupling injection rates using posteriors
    for p in pars_uncert:
        quadruple_injection(ax,p,ts[-1],Cs[-1],lw=0.4,alpha=0.2,label='_nolabel_')
    ax.plot([],[],'m-',label='Quadruple injections')
    
    ax.set_xlabel('Year [A.D.]') # sets x axis label
    ax.set_ylabel('CO2 Concentration [wt%]') # set y axis label
    
    # adds title and legend and shows plot
    ax.set_title('Ohaaki LP concentration model: scenario forecasts with uncertainty')
    ax.legend()
    plt.show()


def main():
    # calibrates parameters
    pars = calibrate_concentration_model([5.88e+03, 1.92e-03,  1.41e-01, 8.80e-04, 4.11e-01,  6.17e+00])
    # plots what-ifs
    plot_predictions_uncert(pars)

if __name__ == "__main__":
    main()
