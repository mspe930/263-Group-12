from pressure_model import pressure_ode, solve_pressure_ode
import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt, scale

def obj_dir(obj, theta, model=None):
    """ Compute a unit vector of objective function sensitivities, dS/dtheta.

        Parameters
        ----------
        obj: callable
            Objective function.
        theta: array-like
            Parameter vector at which dS/dtheta is evaluated.
        
        Returns
        -------
        s : array-like
            Unit vector of objective function derivatives.

    """
    # empty list to store components of objective function derivative 
    s = np.zeros(len(theta))
    
    # compute objective function at theta
    s0 = obj(theta)

    # amount by which to increment parameter
    dtheta = callibration_sensitivity*1.e-4
    
    # for each parameter
    for i in range(len(theta)):

        # basis vector in parameter direction 
        eps_i = np.zeros(len(theta))
        eps_i[i] = 1.
        
        # compute objective function at incremented parameter
        si = obj(theta+dtheta*eps_i)

        # compute objective function sensitivity
        s[i] = (si-s0)/dtheta

    # return sensitivity vector
    return s

def step(theta0, s, alpha):
    """ Compute parameter update by taking step in steepest descent direction.

        Parameters
        ----------
        theta0 : array-like
            Current parameter vector.
        s : array-like
            Step direction.
        alpha : float
            Step size.
        
        Returns
        -------
        theta1 : array-like
            Updated parameter vector.
    """
    # compute new parameter vector as sum of old vector and steepest descent step
    theta1 = theta0 - alpha*s
    
    return theta1


def line_search(obj, theta, s):
    """ Compute step length that minimizes objective function along the search direction.

        Parameters
        ----------
        obj : callable
            Objective function.
        theta : array-like
            Parameter vector at start of line search.
        s : array-like
            Search direction (objective function sensitivity vector).
    
        Returns
        -------
        alpha : float
            Step length.
    """
    length = 0.01*callibration_sensitivity

    # initial step size
    alpha = 0.
    # objective function at start of line search
    s0 = obj(theta)
    # anonymous function: evaluate objective function along line, parameter is a
    sa = lambda a: obj(theta-a*s)
    # compute initial Jacobian: is objective function increasing along search direction?
    # was 0.01
    j = (sa(length)-s0)/length
    # iteration control
    N_max = 500
    N_it = 0
    # begin search
        # exit when (i) Jacobian very small (optimium step size found), or (ii) max iterations exceeded
        # was absj < 1.e-5
    while abs(j) > 1.e-8 and N_it<N_max:
        # increment step size by Jacobian
        alpha += -j
        # compute new objective function
        si = sa(alpha)
        # compute new Jacobian
        j = (sa(alpha+length)-si)/length
        # increment
        N_it += 1
    # return step size
    return alpha


def pressure_misfit(theta,model=None):

    ts_model,Ps_model = solve_pressure_ode(f=pressure_ode,t0=t0,t1=t1,dt=dt,P0=P0,pars=theta)
    f = interp1d(ts_model,Ps_model,bounds_error=False)
    Ps_model = f(ts_data)

    misfit = 0.
    for i in range(len(ts_data)):
        misfit += (Ps_data[i] - Ps_model[i])**2
        # was 1.e-5
    misfit = misfit*callibration_sensitivity*1.e-5
    return misfit

def main():
    ###
    # keep this at 1
    global callibration_sensitivity
    callibration_sensitivity = 1.
    ###

    global ts_data, Ps_data

    data = np.genfromtxt('cs_p.txt',dtype=float,delimiter=', ',skip_header=1).T
    ts_data = data[0,:]
    Ps_data = data[1,:]

    global t0, t1, dt, P0
    t0 = ts_data[0]
    t1 = ts_data[-1]
    dt = 0.2
    P0 = Ps_data[0]

    # theta0 = (M0, a, b, c, d, P0)
    theta0 = np.array([5.00000000e+03,  4.97642037e-03,  4.99956037e-01, -1.61422003e-03,
  5.00000000e-01,  6.16997956e+00])
    s0 = obj_dir(obj=pressure_misfit,theta=theta0)
    
    theta_all = [theta0]
    s_all = [s0]

    N_max = 5
    N_it = 0

    while N_it < N_max:
        # uncomment line below to implement line search (TASK FIVE)
        alpha = line_search(pressure_misfit, theta_all[-1], s_all[-1])
        
        # update parameter vector 
        # **uncomment and complete the command below**
        theta_next = step(theta0,s_all[N_it],alpha)
        theta_all.append(theta_next) 	# save parameter value for plotting
        
        # compute new direction for line search (thetas[-1]
        # **uncomment and complete the command below**
        s_next = obj_dir(pressure_misfit,theta_next)
        s_all.append(s_next) 			# save search direction for plotting
        
        # compute magnitude of steepest descent direction for exit criteria
        N_it += 1
        # restart next iteration with values at end of previous iteration
        theta0 = 1.*theta_next
        s0 = 1.*s_next

        print('Running Optimum: ', theta_all[-1])
        print('Number of iterations so far: ', N_it)
    
    print('\n'*2)
    print('Optimum: ', theta_all[-1])
    print('Number of iterations needed: ', N_it)



if __name__ == "__main__":
    main()