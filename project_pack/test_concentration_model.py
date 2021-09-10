# check c_dash and concentration_ode functions works for the concentration model
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
