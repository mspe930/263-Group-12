import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pressure_model import *

def pressure_model(t,M0,a,b,c,d,P0):
    ''' Uses our pressure model to compute the reservoir pressure at a list of time data given the model parameters.

        Parameters 
        ----------
        t : array-like
            List of time data to compute the reservoir pressure at.
        M0 : float
            Parameter - initial mass of CO2 in reservoir. This is unused in the pressure model but is required for
            consistency with our concentration model.
        a : float
            Source/sink strength lumped parameter.
        b : float
            Recharge strength lumped parameter.
        c : float
            Slow drainage strength lumped parameter.
        d : float
            Diffusion strength lumped parameter. Unused in pressure model but required for consistency with our concentration model.
        P0 : float
            Parameter - initial reservoir pressure.
    
    '''
    # store input parameters as an array
    pars = np.array([M0,a,b,c,d,P0])
    # use pressure model to compute pressure at a range of times
    ts,Ps = solve_pressure_ode(f=pressure_ode,t0=ts_data[0],t1=t[-1],dt=0.2,P0=Ps_data[0],pars=pars)
    # interpolate solved pressures to the list of given time data
    f = interp1d(ts,Ps,kind='linear',bounds_error=False)
    Ps = f(t)
    # return list of pressures at the given list of time data 
    return Ps

def compute_residuals(Ps):
    ''' Compute the residuals of a list of modelled pressures.

        Parameters
        ----------
        Ps : array-like
            List of modelled pressures. These must correspond to times at which the pressure was measured 
            i.e. there should be a one-to-one correspondence between Ps and the list of measured pressures.
        
        Returns
        -------
        residuals : array-like
            List of pressure residuals. 
    
    '''
    # initialise list of residuals
    residuals = np.zeros(len(ts_data))

    # loop through each data point
    for i in range(len(ts_data)):
        # compare the modelled and measured pressures
        residuals[i] = Ps[i] - Ps_data[i]
    # return list of residuals
    return residuals

def main():
    # define global variables for measured data
    global ts_data, Ps_data
    # read pressure data from file
    data = np.genfromtxt('cs_p.txt',dtype=float,delimiter=', ',skip_header=1).T
    # store lists of data as global variables
    ts_data = data[0,:]
    Ps_data = data[1,:]

    # INITIAL PARAMETER GUESS of the form (M0, a, b, c, d, P0)
    pars0 = [5.e+03, 2.5e-3,  3.e-01, 8.e-04, 5.e-01,  6.17e+00]

    # set bounds for parameter estimation
    # P0 is fixed for now so has a very small bounding radius +/- 0.01 MPa
    bounds = ([-1e8]*5 + [Ps_data[0]-0.01], [1e8]*5 + [Ps_data[0]+0.01])

    # using gradient descent method to compute the best fit parameters
    p,_ = curve_fit(pressure_model,ts_data,Ps_data,p0=pars0,bounds=bounds)

    # print the estimated lumped parameters a, b and c
    print("Best fit parameter estimation for pressure model: ")
    print(" "*5, end="")
    print("a = {:1.2e}".format(p[1]))
    print(" "*5, end="")
    print("b = {:1.2e}".format(p[2]))
    print(" "*5, end="")
    print("c = {:1.2e}".format(p[3]))

    # whether to plot the residuals of the fitted model
    plotResiduals = False

    if plotResiduals:
        # solve for the pressures using the best fit parameters
        ts,Ps = solve_pressure_ode(f=pressure_ode,t0=ts_data[0],t1=ts_data[-1],dt=0.2,P0=Ps_data[0],pars=p)
        # interpolate pressure at the measured time data points only
        f = interp1d(ts,Ps)
        Ps = f(ts_data)
        # compute the residuals of the model
        residuals = compute_residuals(Ps)
        
        # plot the residuals 
        f,ax = plt.subplots(1,1)
        ax.plot(ts_data,residuals,'ro',markerSize=8,markerfacecolor='none',label='Fitted Model:\n a = 1.92-03\n b = 1.41e-01\n c = 8.80e-04')
        ax.set_xlabel('Year of observation [A.D.]')
        ax.set_ylabel('Pressure residuals [MPa]')
        ax.set_title('Residuals plot of best fit LP pressure model')
        ax.legend()
        plt.show()


    return p[1],p[2],p[3],p[5]
    
if __name__ == "__main__":
    main()