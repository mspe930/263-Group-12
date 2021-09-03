import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from pressure_model import *

def c_dash(c,c0,P,P0):
    ''' Returns the piece-wise C' term in the CO2 concentration ODE.

        Parameters
        ----------
        c : float
            Concentration of CO2 in reservoir.
        c0 : float
            Ambient concentration of CO2 in reservoir.
        P : float
            Pressure of reservoir.
        P0 : float
            Ambient pressure of reservoir.
        
        Returns
        --------
        c_dash : float
            The term C' in the ODE.
    '''

    if P > P0:
        # computes C' if reservoir pressure is above ambient
        c_dash = c
    else:
        # else computes C' if reservoir pressure is below ambient
        c_dash = c0
    # returns C' term
    return c_dash


def concentration_ode(t, C, C0, P, qco2, M0, a, b, c, d, P0):
    ''' Returns time derivative of CO2 concentration in the reservoir, for given parameters.

        Parameters
        ----------
        t : float
            Independent variable, time.
        C : float
            Dependent variable, CO2 concentration.
        C0 : float
            Ambient CO2 concentration.
        P : float
            Pressure of reservoir.
        qco2 : float
            Rate of CO2 injection. 
        M0 : float
            Initial mass of CO2 in reservoir.
        a : float
            Source/sink strength lumped parameter.
        b : float
            Recharge strength lumped parameter.
        c : float
            Slow drainage strength lumped parameter.
        d : float
            Diffusion strength lumped parameter.
        P0 : float
            Ambient pressure of reservoir.
        
        Returns
        -------
        dCdt : float
            Time derivative of CO2 concentation.
    '''

    Cdash = c_dash(C,C0,P,P0)                # compute C' term in ODE
    termOne = (1-C)*qco2/M0                  # compute first term of ODE
    termTwo = -1*b*(P-P0)*(Cdash-C)/(a*M0)  # compute second term of ODE
    termThree = -1*d*(C-C0)                  # compute third term of ODE

    # sum all terms to find time derivative of CO2 concentration
    dCdt = termOne + termTwo + termThree
    return dCdt

def interpolate_injection(ts):
    ''' Reads CO2 mass injection rates and interpolates this to a given vector of times.

        Parameters
        ---------
        ts : array-like
            Vector of times to interpolate at.

        Returns
        -------
        qs : array-like
            Vector of CO2 mass injection rates for given times in ts.
    '''
    # reads injection data from file
    injection_data = np.genfromtxt('cs_c.txt',dtype=float,delimiter=', ',skip_header=1).T
    # first row of data is times
    ts_data = injection_data[0,:]
    # second row of data is corresponding injection rates
    qco2_data = injection_data[1,:]

    # create interpolation function for data
    f = interp1d(ts_data,qco2_data,kind='linear',fill_value=(0.,0.),bounds_error=False)
    # interpolates mass injection rates for given times
    qs = f(ts)
    return qs

def solve_concentration_ode(f,t0,t1,dt,C0,pars=[]):
    ''' Solves CO2 concentration ODE numerically using the Improved Euler Method.

        Parameters
        ----------
        f : callable
            Function that returns time derivative of CO2 concentration, given parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Step size length.
        C0 : float
            Initial concentration of solution.
        P0 : float
            Ambient pressure of reservoir.
        pars : array-like
            List of parameters passed to f.
    
        Returns
        -------
        ts : array-like
            Independent variable (time) solution vector.
        cs : array-like
            Dependent variable (concentration) solution vector.

        Notes
        -----
        NEED TO EXPLAIN ORDER OF INPUTS TO f
    '''

    # compute number of steps taken for IEM
    npoints = int((t1-t0)/dt + 1)
    # initialise vector of times and concentrations of solution
    ts = np.linspace(t0,t1,npoints,endpoint=True)
    cs = np.zeros(npoints)
    cs[0] = C0

    # find CO2 injection rates for each time
    qs = interpolate_injection(ts)
    # solve for pressures 
    ts,Ps = solve_pressure_ode(f=pressure_ode,t0=t0,t1=t1,dt=dt,P0=pars[-1],pars=pars)

    for i in range(npoints-1):
        f0 = f(ts[i],cs[i],C0,Ps[i],qs[i],*pars)
        f1 = f(ts[i+1],cs[i]+dt*f0,C0,Ps[i],qs[i+1],*pars)
        cs[i+1] = cs[i] + 0.5*dt*(f0+f1)
    
    return ts,cs
    
if __name__ == "__main__":
    cs_data = np.genfromtxt('cs_cc.txt',dtype=float,delimiter=', ',skip_header=1).T
    ts_data = cs_data[0,:]
    tmin = ts_data[0]
    tmax = ts_data[-1]
    cs_data = cs_data[1,:]
    c0 = cs_data[0]

    ####################################################
    ## ALL OF THESE CONSTANTS CHANGE OUR MODEL #########
    M0 = 5.e3           # note this is mass of co2 not the mass of our entire reservoir
    a = 8.e-5
    b = 1.e-2
    c = 7.e-3
    d = 5.e-1
    P0 = 6.17e+00
    ####################################################

    pars = [M0,a,b,c,d,P0]

    ts_model,cs_model = solve_concentration_ode(f=concentration_ode,t0=tmin,t1=tmax,dt=0.05,C0=cs_data[0],pars=pars)

    f,ax = plt.subplots(1,1)
    ax.plot(ts_model,cs_model,'r-',label='Fitted Model')
    ax.plot(ts_data,cs_data,'kx',label='Measured Data')
    ax.legend()
    ax.set_title('Comparison of modelled CO2 concentration versus measured data over time in the Ohaaki geothermal reservoir')
    ax.set_xlabel('Year of measurement [A.D.]')
    ax.set_ylabel('Concentration of CO2 [wt%]')
    plt.show()