# import modules
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
    s0 = obj(theta, model) 

    # amount by which to increment parameter
    dtheta = 1.e-2 
    
    # for each parameter
    for i in range(len(theta)):
        # basis vector in parameter direction 
        eps_i = np.zeros(len(theta))
        eps_i[i] = 1.e-3
        
        # compute objective function at incremented parameter
        si = obj(theta+dtheta*eps_i, model)

        # compute objective function sensitivity
        s[i] = (si - s0) / dtheta

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
    theta1 = theta0 - (s * alpha)
    
    return theta1


def line_search(obj, theta, s, model):
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
    alpha = 1.e-6
    # objective function at start of line search
    s0 = obj(theta, model)
    # anonymous function: evaluate objective function along line, parameter is a
    sa = lambda a: obj(theta-a*s, model)
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

def grad_descent(obj, theta0, alpha, N_max, model, plot_f):
    theta_all = [theta0]
    s_all = [obj_dir(obj, theta_all[-1], model)]
    N_it = 0
    lowest = (9999999., [])
    #while obj(theta_all[-1], model) > 200:
    while N_max - N_it > 0:
        # Line search algorithim
        #alpha = line_search(obj, theta_all[-1], s_all[-1], model)
         
        #if N_it % 1000:
        #    import random
        #    theta_all[-1][random.randint(0,2)] += alpha

        # update parameter vector 
        theta_next = step(theta_all[-1], s_all[-1], alpha)
        theta_all.append(theta_next) 	# save parameter value for plotting

        # compute new direction for line search (thetas[-1]
        s_next = obj_dir(obj, theta_next, model)
        s_all.append(s_next) 			# save search direction for plotting
        
        # compute magnitude of steepest descent direction for exit criteria
        N_it += 1

        theta0 = 1.*theta_next
        s0 = 1.*s_next
        cost = obj(theta_all[-1], model)

        plot_f(theta_all[-1], model)

    print('Number of iterations needed: ', N_it)

    print("Change:", theta_all[0], "->", theta_all[-1])
    print("Objective change: ", obj(theta_all[0], model), "->", obj(theta_all[-1], model))
    return theta_all[-1]