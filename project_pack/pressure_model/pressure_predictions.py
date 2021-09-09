import numpy as np
from matplotlib import pyplot as plt
from pressure_calibration import *
from pressure_model import *

def solve_pressure_custom(qs,f,t0,t1,dt,P0,pars=[]):
    ''' Solves pressure ODE numerically given a custom list of mass flow rates (not read from file).
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
        P0 : float
            Initial pressure value of solution.
        pars : array-like
            List of lumped parameters passed to function f.
        Returns
        -------
        ts : array-like
            Independent variable solution vector.
        Ps : array-like
            Dependent variable solution vector.
        
        Notes
        -----
        Assume that the ODE function f takes the following inputs, in order:
            (i)   independent variable
            (ii)  dependent variable
            (iii) forcing term, q
            (iv)  time derivative of forcing term, dq/dt
            (v)   all other parameters
        This function was designed to be identical to the solve_pressure_ode function in 
        pressure_model.py but with a custom input of mass flow rates qs. 
    
    '''
    npoints = int((t1-t0)/dt + 1)
    # initialise vector of times and pressures of solution
    ts = np.linspace(t0,t1,npoints,endpoint=True)
    Ps = np.zeros(npoints)
    Ps[0] = P0

    # find time derivative of net mass flow rate for each time
    qs = np.array([qs[0]]*len(ts))
    dqdts = compute_dqdt(ts,qs)

    # loop through each step in the IEM
    for i in range(npoints-1):
        # compute f0 term 
        f0 = f(ts[i],Ps[i],qs[i],dqdts[i],*pars)
        # compute f1 term
        f1 = f(ts[i+1],Ps[i]+dt*f0,qs[i+1],dqdts[i+1],*pars)
        # find next step of pressure
        Ps[i+1] = Ps[i] + 0.5*dt*(f0+f1)

    # return time and pressure solution vectors
    return ts,Ps


def half_injection(ax,pars,tend,Pend):
    ''' Solves our pressure model to predict the outcome when the injection rate is halved, and plots this solution
        on a given axis.
        Parameters
        ----------
        ax : matplot.lib object
            Axis to plot the predicted pressure evolution.
        pars : array-like
            List of lumped parameters passed to the pressure model.
        tend : float
            Present time.
        Pend : float
            Pressure of reservoir at last measurement (at present time).
        
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
    # creates list of future injection rates - assuming no change in injection rates from now 
    qs_inj = np.array([0.5*qs_inj[-1]]*N)
    # computes list of net mass flow rates 
    qs = qs_prod - qs_inj

    # solves pressure model using custom list of injection rates 
    ts,Ps = solve_pressure_custom(qs=qs,f=pressure_ode,t0=tmin,t1=tmax,dt=0.2,P0=Pend,pars=pars)
    # plots pressure evolution on axis 
    ax.plot(ts,Ps,'y-',label='Half injection')


def no_changes_injection(ax,pars,tend,Pend):
    ''' Solves our pressure model to predict the outcome when no changes are made to the injection rate, and plots this solution
        on a given axis.
        Parameters
        ----------
        ax : matplot.lib object
            Axis to plot the predicted pressure evolution.
        pars : array-like
            List of lumped parameters passed to the pressure model.
        tend : float
            Present time.
        Pend : float
            Pressure of reservoir at last measurement (at present time).
        
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
    # creates list of future injection rates - assuming no change in injection rates from now 
    qs_inj = np.array([qs_inj[-1]]*N)
    # computes list of net mass flow rates 
    qs = qs_prod - qs_inj

    # solves pressure model using custom list of injection rates 
    ts,Ps = solve_pressure_custom(qs=qs,f=pressure_ode,t0=tmin,t1=tmax,dt=0.2,P0=Pend,pars=pars)
    # plots pressure evolution on axis 
    ax.plot(ts,Ps,'b-',label='No change')


def quadruple_injection(ax,pars,tend,Pend):
    ''' Solves our pressure model to predict the outcome when the injection rates are quadrupled, and plots this solution
        on a given axis.
        Parameters
        ----------
        ax : matplot.lib object
            Axis to plot the predicted pressure evolution.
        pars : array-like
            List of lumped parameters passed to the pressure model.
        tend : float
            Present time.
        Pend : float
            Pressure of reservoir at last measurement (at present time).
        
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
    qs_inj = np.array([4*qs_inj[-1]]*N)
    # computes list of net mass flow rates 
    qs = qs_prod - qs_inj

    # solves pressure model using custom list of injection rates 
    ts,Ps = solve_pressure_custom(qs=qs,f=pressure_ode,t0=tmin,t1=tmax,dt=0.2,P0=Pend,pars=pars)

    # plots pressure evolution on axis 
    ax.plot(ts,Ps,'m-',label='Quadruple injections')


def double_injection(ax,pars,tend,Pend):
    ''' Solves our pressure model to predict the outcome when the injection rates are doubled, and plots this solution
        on a given axis.
        Parameters
        ----------
        ax : matplot.lib object
            Axis to plot the predicted pressure evolution.
        pars : array-like
            List of lumped parameters passed to the pressure model.
        tend : float
            Present time.
        Pend : float
            Pressure of reservoir at last measurement (at present time).
        
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
    qs_inj = np.array([2*qs_inj[-1]]*N)
    # computes list of net mass flow rates 
    qs = qs_prod - qs_inj

    # solves pressure model using custom list of injection rates 
    ts,Ps = solve_pressure_custom(qs=qs,f=pressure_ode,t0=tmin,t1=tmax,dt=0.2,P0=Pend,pars=pars)

    # plots pressure evolution on axis 
    ax.plot(ts,Ps,'g-',label='Double injections')


def plot_predictions(pars):
    ''' Plots the predicted pressure evolutions of all possible scenarios alongside the fitted model and measured data, given 
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
    ts_data,Ps_data = fetch_pressure_data()
    ax.plot(ts_data,Ps_data,'ro',label='Measured data')
    
    # solves fitted model and plots it alongside data 
    ts,Ps = solve_pressure_ode(pressure_ode,ts_data[0],ts_data[-1],0.2,Ps_data[0],pars)
    ax.plot(ts,Ps,'k-',label='Best fit model')

    # plots the predicted pressure evolution when quadrupling injection rate 
    quadruple_injection(ax,pars,ts[-1],Ps[-1])
    # plot the predicted pressure evolution when doubling injection rate
    double_injection(ax,pars,ts[-1],Ps[-1])
    # plots the predicted pressure evolution with no change in current injection rate
    no_changes_injection(ax,pars,ts[-1],Ps[-1])
    # plots the predicted pressure evolutions when halving injection rate
    half_injection(ax,pars,ts[-1],Ps[-1])

    ax.legend() # add legend
    plt.show()  # show plot


def main():
    p = calibrate_pressure_model([5.e+03, 2.5e-3,  3.e-01, 8.e-04, 5.e-01,  6.17e+00])
    plot_predictions(p)

if __name__ == "__main__":
    main()  