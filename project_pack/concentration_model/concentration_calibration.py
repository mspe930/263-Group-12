import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from concentration_model import *
from pressure_calibration import calibrate_pressure_model

def concentration_model(t,M0,a,b,c,d,P0):
    ''' Uses the concentation model to compute the CO2 concentration at a list of time data given the model parameters.

        Parameters 
        ----------
        t : array-like
            List of time data to compute the reservoir pressure at.
        M0 : float
            Parameter - initial mass of CO2 in reservoir.
        a : float
            Source/sink strength lumped parameter.
        b : float
            Recharge strength lumped parameter.
        c : float
            Slow drainage strength lumped parameter.
        d : float
            Diffusion strength lumped parameter. 
        P0 : float
            Parameter - initial reservoir pressure. 
        
        Returns
        -------
        Cs : array-like
            List of CO2 concentrations for parameters and paired times. 
    
    '''
    # store input parameters as an array
    pars = np.array([M0,a,b,c,d,P0])
    # read concentration data
    ts_data,Cs_data = fetch_concentration_data()
    # use concentration model to compute CO2 concentrations at a range of times
    ts,Cs = solve_concentration_ode(f=concentration_ode,t0=ts_data[0],t1=t[-1],dt=0.2,C0=Cs_data[0],pars=pars)
    # interpolate solved concentrations for given list of times 
    f = interp1d(ts,Cs,kind='linear',bounds_error=False)
    Cs = f(t)
    # return list of concentrations for given list time data 
    return Cs


def compute_residuals(Cs):
    ''' Compute the residuals of a list of modelled CO2 concentrations.

        Parameters
        ----------
        Cs : array-like
            List of modelled CO2 concentrations. These must correspond to times at which the concentration was measured 
            i.e. there should be a one-to-one correspondence between Cs and the list of measured concentrations.
        
        Returns
        -------
        residuals : array-like
            List of concentration residuals. 
    
    '''
    # read concentration data 
    ts_data,Cs_data = fetch_concentration_data()
    # initialise list of residuals 
    residuals = np.zeros(len(ts_data))

    # loop through each data point
    for i in range(len(ts_data)):
        # compare measured and modelled concentrations
        residuals[i] = Cs[i] - Cs_data[i]
    # return list of residuals 
    return residuals


def plot_concentration_residuals(pars):
    ''' Creates a residual plot of a concentration model given a list of model parameters

        Parameters
        ----------
        pars : array-like
            List of model parameters in the form (M0, a, b, c, d, P0)
        
        Returns 
        -------
        None
    '''
    # read concentration data 
    ts_data,Cs_data = fetch_concentration_data()
    # solve concentration model using given parameters
    ts,Ps = solve_concentration_ode(f=concentration_ode,t0=ts_data[0],t1=ts_data[-1],dt=0.2,C0=Cs_data[0],pars=pars)
    
    # interpolate CO2 concentration at the measured data points only 
    f = interp1d(ts,Ps,kind='linear',bounds_error=False)
    Cs = f(ts_data)
    # compute residuals of model
    residuals = compute_residuals(Cs)

    # plot the residuals 
    f,ax = plt.subplots(1,1)
    ax.plot(ts_data,residuals,'ro',markerSize=8,markerfacecolor='none',label='Fitted Model:\n M0 = {:1.2e}\n a = {:1.2e}\n b = {:1.2e}\n c = {:1.2e}\n d = {:1.2e}'.format(*pars[:-1]))
    ax.set_xlabel('Year of observation [A.D.]')
    ax.set_ylabel('Residuals [MPa]')
    ax.set_title('Residuals plot of fitted LP CO2 concentration model')
    ax.legend()
    plt.show()


def calibrate_concentration_model(pars0):
    ''' Calibrates our concentration model using a steepest descent algorithm and a sum-of-squares misfit.

        Parameters
        ----------
        pars0 : array-like
            List of model parameters (initial guess).
        
        Returns
        -------
        pars : array-like
            List of parameters of our calibrated model (give a best fit).
        
        Notes
        -----
        Both pars0 and pars are a vector of the form (M0, a, b, c, d, P0).
    '''
    # reads data off file
    ts_data,Cs_data = fetch_concentration_data()
    # calibrate a, b and c using the pressure model calibration on pressure_calibration.py
    p = calibrate_pressure_model(pars0)

    # define lower bounds for calibration
    lwr_bnds = [0., .999*p[1], .999*p[2], .999*p[3], 0., .999*p[5]]
    # define upper bounds for calibration
    upr_bnds = [10.e10, 1.001*p[1], 1.001*p[2], 1.001*p[3], 10.e10, 1.001*p[5]]
    # combine lower and upper bounds 
    bounds = (lwr_bnds, upr_bnds)

    # using gradient descent method to compute the best fit parameters
    p,_ = curve_fit(concentration_model,ts_data,Cs_data,p0=p,bounds=bounds)

    # print the estimated parameters M0 and d (other parameters are already printed when finding p)
    print("\nAdditional parameter estimations for concentration model:")
    print(" "*5, end="")
    print("d = {:1.2e}".format(p[4]))
    print(" "*4, end="")
    print("M0 = {:1.2e}\n".format(p[0]))

    # return list of parameters of calibrated model
    return p

def main():
    # initial guess of parameters
    pars0 = [8.e+03, 2.5e-3,  3.e-01, 8.e-04, 5.e-01,  6.17e+00]
    # find best fit parameters
    pars = calibrate_concentration_model(pars0)
    # plot best fit model
    plot_concentration(pars)
    # plot residuals of best fit model
    plot_concentration_residuals(pars)

if __name__ == "__main__":
    main()