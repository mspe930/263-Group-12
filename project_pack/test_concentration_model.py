# check c_dash, concentration_ode, and solve_concentration_ode functions works for the concentration model
# by running unit tests that compares output to a by-hand example
# run pytest to automatically run unit tests

import numpy  as np
from concentration_model import *
from pressure_model import *
from numpy.linalg import norm

def test_c_dash():
    """ Tests c_dash function in concentration_model
    Checks for three instances: P > P0, P = P0, P < P0
    """
    c = 3.07*10**(-2)
    C0 = 3.0
    P = 5.38
    P0 = 6.0

    C = c_dash(c, C0, P, P0)
    assert norm(C - 3.0) < 1.e-10

    P = 6.0
    C = c_dash(c, C0, P, P0)
    assert norm(C - 3.0) < 1.e-10

    P = 10
    C = c_dash(c, C0, P, P0)
    assert norm(C - 3.07*10**(-2)) < 1.e-10

    c = 0
    C0 = 0
    P = 0
    P0 = 0
    C = c_dash(c, C0, P, P0)
    assert norm(C - 0) < 1.e-10
    
def test_concentration_ode():
    """
    Test  if function concentration_ode works as expected, 
    Checks variables = 0, term 1, 2, 3
    """
    # Need t, C, C0, P, q, M0, a, b, c, d, P0
    [t, c] = [0, 0]
    [C, C0, P, q, M0, a, b, d, P0] = [0, 0, 0, 0, 1, 1, 0, 0, 0]
    # neither a nor M0 can be zero else dividing by zero
    dCdt = concentration_ode(t, C, C0, P, q, M0, a, b, c, d, P0)
    assert norm(dCdt - 0) < 1.e-10
    
    # check each term
    [C, q, M0] = [0.5, 8, 2]
    dCdt = concentration_ode(t, C, C0, P, q, M0, a, b, c, d, P0)
    assert norm(dCdt - 2) < 1.e-10

    [C, C0, P, q, M0, a, b, d, P0] = [2, 0, 1, 0, 2, 6, 3, 0, 2]
    dCdt = concentration_ode(t, C, C0, P, q, M0, a, b, c, d, P0)
    ans = -0.5
    assert norm(dCdt - ans) < 1.e-10

    [C, C0, P, q, M0, a, b, d, P0] = [1.2, 1, 0, 0, 1, 1, 0, -4, 0]
    dCdt = concentration_ode(t, C, C0, P, q, M0, a, b, c, d, P0)
    assert norm(dCdt - 0.8) < 1.e-10
    
def test_solve_concentration_ode():
    """
    Test if solve_concentration_ode works as intended
    """
    ts,cs = solve_concentration_ode(f=concentration_ode,t0=2000,t1=2020,dt=1,C0=0.1,pars=[1000,0.4,1,0.002,2,15],dash_ignore=True)

    tsans = np.array([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
    csans = np.array([0.1,0.0998,0.1006,0.0976,0.0951,0.0932,0.0922,0.0915,0.0916,0.0940,0.0912,0.0855,0.0798,0.0755,0.0717,0.0693,0.0687,0.0658,0.0623,0.0594,0.0365])
    assert norm(ts - tsans) < 1.e-10
    assert norm(cs - csans) < 1.e-3
