import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

def fetch_pressure_data():
    ''' Reads the pressure data on the cs_p.txt file.
        Parameters
        ----------
        None
        Returns 
        -------
        ts_data : array-like
            Vector of times.
        Ps_data : array-like
            Vector of reservoir pressure readings.
    '''
    # read pressure data from file
    data = np.genfromtxt('cs_p.txt',dtype=float,delimiter=', ',skip_header=1).T
    # seperates data into times and pressures
    ts_data = data[0,:]
    Ps_data = data[1,:]
    # returns vectors of data
    return ts_data,Ps_data

def pressure_ode(t, P, q, dqdt, M0, a, b, c, d, P0):
    ''' Returns time derivative of reservoir pressure, dP/dt, for given parameters.
        
        Parameters
        ----------
        t : float
            Independent variable, time.
        P : float
            Dependent variable, reservoir pressure.
        q : float
            Mass flow rate into reservoir. 
        dqdt : float
            Rate of change of mass flow rate into reservoir.
        M0 : float
            (Unused: required for consistency).
            The total mass of CO2 in the reservoir. 
        a : float
            Source/sink strength lumped parameter.
        b : float
            Recharge strength lumped parameter.
        c : float
            Slow drainage strength lumped parameter.
        d : float
            Diffusion strength lumped parameter.
        P0 : float
            Initial pressure of reservoir. Treated a fixed constant until the uncertainty analysis 
            (in which it is treated as another model parameter).
    
        Returns
        -------
        dPdt : float
            Time derivative of reservoir pressure (dependent variable).
        Notes
        -----
        Parameters must be passed in the above order.
    '''
    # calculates dP/dt using ODE
    dPdt = -1*a*q - b*(P-P0) - c*dqdt
    return dPdt


def interpolate_mass_flow(ts):
    ''' Reads injection and production rates, interpolates this data, and calculates
        the net mass flow into the reservoir. 
        Parameters
        ----------
        ts : array-like
            Vector of times to interpolate the mass flow rate.
        
        Returns
        -------
        qs : array-like
            Vector of net mass flow rates for given times. 
    '''
    # reads injection rates of CO2
    qco2 = np.genfromtxt('cs_c.txt',dtype=float,delimiter=', ',skip_header=1).T
    # reads extraction rates of water
    qwater = np.genfromtxt('cs_q.txt',dtype=float,delimiter=', ',skip_header=1).T

    # interpolates injection rates for given times
    fco2 = interp1d(qco2[0,:],qco2[1,:],kind='linear',fill_value=(0.,0.),bounds_error=False)
    qco2_fit = fco2(ts)
    # interpolates extraction rates for given times
    fwater = interp1d(qwater[0,:],qwater[1,:],kind='linear',fill_value=(0.,0.),bounds_error=False)
    qwater_fit = fwater(ts)

    # computes net mass flow rates = injection - extraction
    qs = qwater_fit - qco2_fit
    # returns vector of net mass flow rates
    return qs


def compute_dqdt(ts, qs):
    ''' Computes the time derivative of net mass flow rates, dq/dt. 
        Parameters
        ----------
        ts : array-like
            Vector of times to compute dq/dt at.
        qs : array-like
            Vector of net mass flow rates. 
        
        Returns
        -------
        dqdts : array-like
            Vector of dq/dt values for given times.
        
        Notes
        -----
        The vectors ts and qs must be paired measurements, i.e. qs[i] correponds to the 
        mass flow rate measured at time ts[i], for all i.
        The algorithm used requires a very small step size for accuracy. Hence, the data 
        should be interpolated before passing it into this function.
    '''
    # initialise array of dq/dt values
    dqdts = np.zeros(len(ts))
    dqdts[-1] = 0.
    
    # loop over each paired measurement
    for i in range(len(ts)-1):
        # computes dq/dt at each measurement time
        deltat = ts[i+1]-ts[i]
        dqdts[i] = (qs[i+1] - qs[i])/deltat
    
    return dqdts 


def solve_pressure_ode(f,t0,t1,dt,P0,pars=[],benchmarkq=False):
    ''' Solves pressure ODE numerically using the Improved Euler Method.
        Parameters
        ----------
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
        benchmarkq : boolean
            True is when benchmarking and taking net mass extraction rate = 150 = const. False otherwise.
        
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
    '''
    # compute number of steps taken for IEM
    npoints = int((t1-t0)/dt + 1)
    # initialise vector of times and pressures of solution
    ts = np.linspace(t0,t1,npoints,endpoint=True)
    Ps = np.zeros(npoints)
    Ps[0] = P0

    # find the net mass flow rate for each time
    if benchmarkq:
        qs = np.array([150.]*len(ts))   # when benchmarking assume q = 150 = const.
    else:
        qs = interpolate_mass_flow(ts)  # when not benchmarking interpolate q

    # find time derivative of net mass flow rate for each time
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

def plot_pressure(pars):
    ''' Plots the LP pressure model against measured data given a list of model parameters.
        Parameters
        ----------
        pars : array-like
            List of model parameters in the form (M0, a, b, c, d, P0).
        
        Returns
        -------
        None
    '''

    # reads pressure measurements from file 
    ts_data,Ps_data = fetch_pressure_data()
    # takes minimum time of measurement 
    tmin = ts_data[0]
    # takes maximum time of measurement
    tmax = ts_data[-1]
    # takes initial measured pressure
    P0 = Ps_data[0]

    # solves our pressure model for the given parameters
    ts_model,Ps_model = solve_pressure_ode(f=pressure_ode,t0=tmin,t1=tmax,dt=0.2,P0=P0,pars=pars)

    # plots our pressure model
    f,ax = plt.subplots(1,1)
    ax.plot(ts_data,Ps_data,'kx',label='Measured Data')
    ax.plot(ts_model,Ps_model,'r-',label='Fitted Model:\n a = {:1.2e}\n b = {:1.2e}\n c = {:1.2e}'.format(pars[1],pars[2],pars[3]))
    ax.set_xlabel('Year of observation [A.D.]')
    ax.set_ylabel('Reservoir pressure [MPa]')
    ax.legend()
    ax.set_title('Fitted LP pressure model')
    plt.show()

def main():
    # initial guess of parameters
    plot_pressure([5.e+03, 2.5e-3,  3.e-01, 8.e-04, 5.e-01,  6.17e+00])


if __name__ == "__main__":
    main()
