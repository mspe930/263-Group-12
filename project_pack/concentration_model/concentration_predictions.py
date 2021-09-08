from concentration_model import *
from concentration_calibration import *
import numpy as np
from matplotlib import pyplot as plt
from pressure_calibration import *
from pressure_model import *
from pressure_predictions import solve_pressure_custom

def solve_concentration_custom(qs,f,t0,t1,dt,C0,pars=[]):
    ''' Solves CO2 concentration ODE numerically given a custom list of mass flow rates (not read from file).

        Parameters
        ----------
        qs : array-like
            Vector of mass flow rates. 
        f : callable
            Function that returns time-derivative of pressure given parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Step size length.
        C0 : float
            Initial concentration of solution.
        pars : array-like
            List of lumped parameters passed to function f.

        Returns
        -------
        ts : array-like
            Independent variable solution vector.
        Cs : array-like
            Dependent variable solution vector.
        
        Notes
        -----
        This function was designed to be identical to the solve_pressure_ode function in 
        pressure_model.py but with a custom input of mass flow rates qs. 
    
    '''
    # compute number of steps taken for IEM
    npoints = int((t1-t0)/dt + 1)
    # initialise vector of times and concentrations of solution
    ts = np.linspace(t0,t1,npoints,endpoint=True)
    Cs = np.zeros(npoints)
    Cs[0] = C0

    # create list of mass flow rates (assumed const.)
    qs = np.array([qs[0]]*len(ts))

    # solve for pressures
    ts,Ps = solve_pressure_custom(qs,pressure_ode,t0,t1,dt,pars[-1],pars)

    # loop through each step in the IEM
    for i in range(npoints-1):
        # compute f0 term 
        f0 = f(ts[i],Cs[i],C0,Ps[i],qs[i],*pars)
        # compute f1 term
        f1 = f(ts[i+1],Cs[i]+dt*f0,C0,Ps[i],qs[i+1],*pars)
        # find next step of concentration
        Cs[i+1] = Cs[i] + 0.5*dt*(f0+f1)

    # return time and concentration solution vectors
    return ts,Cs


def no_changes_injection(ax,pars,tend,Pend):
    pass

def quadruple_injection(ax,pars,tend,Pend):
    pass

def double_injection(ax,pars,tend,Pend):
    pass

def half_injections(ax,pars,tend,Cend):
    ''' Solves our CO2 concentration model to predict the outcome when the injection rates are halved, and plots this solution
        on a given axis.

        Parameters
        ----------
        ax : matplot.lib object
            Axis to plot the predicted pressure evolution.
        pars : array-like
            List of lumped parameters passed to the pressure model.
        tend : float
            Present time.
        Cend : float
            Concentration of CO2 at last measurement (at present time).
        
        Returns
        -------
        None
    '''
    # reads past injection data from file 
    inj_data = np.genfromtxt('cs_c.txt',dtype=float,delimiter=', ',skip_header=1).T
    qs_inj = inj_data[1,:]

    # reads past production data from file 
    prod_data = np.genfromtxt('cs_q.txt',dtype=float,delimiter=', ',skip_header=1).T
    qs_prod = prod_data[1,:]

    # finds start time of prediction (i.e. the real present)
    tmin = tend
    # finds end time of prediction, 20 yrs into the future
    tmax = tmin + 20.
    # creates list of times 
    N = 50
    ts = np.linspace(tmin,tmax,N,endpoint=True)

    # creates list of future production rates - assuming constant from now
    qs_prod = np.array([qs_prod[-1]]*N)
    # creates list of future injection rates - assuming quadruple of current rate 
    qs_inj = np.array([0.5*qs_inj[-1]]*N)
    # computes list of net mass flow rates 
    qs = qs_prod - qs_inj

    # solves concentration model using custom list of injection rates 
    ts,Ps = solve_concentration_custom(qs=qs,f=concentration_ode,t0=tmin,t1=tmax,dt=0.2,C0=Cend,pars=pars)

    # plots pressure evolution on axis 
    ax.plot(ts,Ps,'y-',label='Half injections')


def plot_predictions(pars):
    ''' Plots the predicted CO2 concentration evolutions of all possible scenarios alongside the fitted model and measured data, given 
        a list of model parameters.

        Parameters
        ----------
        pars : array-like
            List of pressure model parameters.

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
    ts,Cs = solve_concentration_ode(concentration_ode,ts_data[0],ts_data[-1],0.2,Cs_data[0],pars)
    ax.plot(ts,Cs,'k-',label='Best fit model')

    # plots the predicted pressure evolution when quadrupling injection rate 
    quadruple_injection(ax,pars,ts[-1],Cs[-1])
    # plot the predicted pressure evolution when doubling injection rate
    double_injection(ax,pars,ts[-1],Cs[-1])
    # plots the predicted pressure evolution with no change in current injection rate
    no_changes_injection(ax,pars,ts[-1],Cs[-1])
    # plots the predicted pressure evolution when halving injection rate
    half_injections(ax,pars,ts[-1],Cs[-1])

    ax.set_ylabel('CO2 concentration [wt%]')   # set y axis label
    ax.set_xlabel('Year [A.D.]')                # set x axis label
    ax.set_title('CO2 concentration LP model: scenario forecasts')   # set title 
    ax.legend() # add legend
    plt.show()  # show plot


def main():
    p = calibrate_concentration_model([5.e+03, 2.5e-3,  3.e-01, 8.e-04, 5.e-01,  6.17e+00])
    plot_predictions(p)

if __name__ == "__main__":
    main()  