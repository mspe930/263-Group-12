from callibration_functions import calc_sensitivity
from pressure_model import *
import numpy as np

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
    dtheta = 1.e-2
    
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
    # initial step size
    alpha = 0.
    # objective function at start of line search
    s0 = obj(theta)
    # anonymous function: evaluate objective function along line, parameter is a
    sa = lambda a: obj(theta-a*s)
    # compute initial Jacobian: is objective function increasing along search direction?
    j = (sa(.01)-s0)/0.01
    # iteration control
    N_max = 500
    N_it = 0
    # begin search
        # exit when (i) Jacobian very small (optimium step size found), or (ii) max iterations exceeded
    while abs(j) > 1.e-5 and N_it<N_max:
        # increment step size by Jacobian
        alpha += -j
        # compute new objective function
        si = sa(alpha)
        # compute new Jacobian
        j = (sa(alpha+0.01)-si)/0.01
        # increment
        N_it += 1
    # return step size
    return alpha
