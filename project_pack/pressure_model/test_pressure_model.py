from pressure_model import pressure_ode
from pressure_model import interpolate_mass_flow
from pressure_model import compute_dqdt
from pressure_model import solve_pressure_ode
import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

#unit tests

#pressure_ode
#testing various values, incl negative and 0 values
odeTest1 = pressure_ode(0,0,0,0,0,0,0,0,0,0)
print(odeTest1)
assert odeTest1 == 0
odeTest2 = pressure_ode(1,1,-30.5,-10.2,10,1,1,1,0,0)
print(odeTest2)
assert odeTest2 == 39.7

#interpolate_mass_flow
#check for values in data to see if function works as intended
intTest1 = interpolate_mass_flow(np.array([1998.51,2010.56]))
print(intTest1)
assert np.all(intTest1 == np.array([126.7, 128.8]))
#check for values not in range
intTest2 = interpolate_mass_flow(np.array([-400,805,200555]))
print(intTest2)
assert np.all(intTest2 == np.array([0,0,0]))
#check single value not in array
intTest3 = interpolate_mass_flow(2000)
print(intTest3)

#compute_dqdt
#testing negative values
comTest1 = compute_dqdt(np.array([1,2,3,4,5]), np.array([-100,-200,-300,400,500]))
print(comTest1)
comTest2 = compute_dqdt(np.array([-4,2,-4,4,-1]), np.array([10.5,0,10.5,40.06,50011.4]))
print(comTest2)
#test real values
comTest3 = compute_dqdt(np.array([1998.51,2010.56]), np.array([126.7, 128.8]))
print(comTest3)
assert np.all(abs(comTest3 - np.array([0.17427386,0])) < 0.001)

#solve_pressure_ode
solTest1 = solve_pressure_ode(f=pressure_ode,t0=0,t1=1,dt=0.1,P0=100,pars=[0,0,0,0,0,0])
print(solTest1)
#test unusual step sizes
solTest2 = solve_pressure_ode(f=pressure_ode,t0=0,t1=10,dt=100,P0=-100,pars=[0,0,0,0,0,0])
print(solTest2)
solTest3 = solve_pressure_ode(f=pressure_ode,t0=0,t1=100,dt=0.1,P0=-100,pars=[1,0,1,10,0,0])
print(solTest3)
