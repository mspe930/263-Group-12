from urllib import parse
import numpy as np
from concentration_model import *
from concentration_predictions import *
from concentration_calibration import concentration_model
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def fo_cov_estimate(pars):
    ''' Generates a sensible covariance array by using the solution simplified version of the concentration
        ODE and the pressure ODE.

        Parameters
        ----------
        pars : array-like
            List of best fit parameters for our concentration model of the form (M0, a, b, c, d, P0).

        Returns
        -------
        cov : array-like
            A sensible covariance matrix estimate.
    ''' 
    # Fetches pressure data 
    ts_data, Ps_data = fetch_pressure_data()

    # Sigma Estimate
    sigma = [.25]*len(ts_data)
    
    # Bounds based on the pressure data 
    bounds = ([-1e8]*5 + [Ps_data[0]-0.01], [1e8]*5 + [Ps_data[0]+0.01])
    p, p_cov = curve_fit(pressure_model, ts_data, Ps_data, p0=pars, bounds=bounds, sigma=sigma)

    # fetches and plots measured data 
    ts_data,Cs_data = fetch_concentration_data()

    # Concentration bounds based on the resturns from pressure_data 
    lwr_bnds = [0., .999*p[1], .999*p[2], .999*p[3], 0., .999*p[5]]
    upr_bnds = [10.e10, 1.001*p[1], 1.001*p[2], 1.001*p[3], 10.e10, 1.001*p[5]]
    bounds = (lwr_bnds, upr_bnds)
    
    # Sigma Estiamte
    sigma = [.25]*len(ts_data)

    # Conc model
    conc_model = lambda t, M0, a, b, c, d, P0: concentration_model(t, M0, a, b, c, d, P0, True)
    p, c_cov = curve_fit(conc_model, ts_data, Cs_data, p0=p, bounds=bounds, sigma=sigma)

    return p_cov + c_cov

if __name__ == "__main__":
    cov = fo_cov_estimate([5.88e+03, 1.92e-03,  1.41e-01, 8.80e-04, 4.11e-01,  6.17e+00])
    print(cov)