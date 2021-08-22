import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

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


def concentration_ode(t, c, c0, P, P0, qco2, M0, a, b, d):
    ''' Returns time derivative of CO2 concentration in the reservoir, for given parameters.

        Parameters
        ----------
        t : float
            Independent variable, time.
        c : float
            Dependent variable, CO2 concentration.
        c0 : float
            Ambient CO2 concentration.
        P : float
            Pressure of reservoir.
        P0 : float
            Ambient pressure of reservoir.
        qco2 : float
            Rate of CO2 injection. 
        M0 : float
            Initial mass of reservoir.
        a : float
            Source/sink strength lumped parameter.
        b : float
            Recharge strength lumped parameter.
        d : float
            Diffusion strength lumped parameter.
        
        Returns
        -------
        dcdt : float
            Time derivative of CO2 concentation.
    '''

    cdash = c_dash(c,c0,P,P0)                # compute C' term in ODE
    termOne = (1-c)*qco2/M0                  # compute first term of ODE
    termTwo = -1*b*(P-P0)*(c_dash-c)/(a*M0)  # compute second term of ODE
    termThree = -1*d*(c-c0)                  # compute third term of ODE

    # sum all terms to find time derivative of CO2 concentration
    dcdt = termOne + termTwo + termThree
    return dcdt

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


