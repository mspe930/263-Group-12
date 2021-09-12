# plots production against pressure(and concentration?)

import numpy as np
from numpy.lib.npyio import genfromtxt, load
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

data = np.genfromtxt('cs_c.txt', delimiter=',', skip_header=1)
ti = data[:, 0]
inj = data[:, 1]
data = np.genfromtxt('cs_q.txt', delimiter=',', skip_header=1)
tp = data[:, 0]
pro = data[:, 1]

ts = np.linspace(tp[0], tp[-1], 55, endpoint=True)

# reads injection rates of CO2
qco2 = np.genfromtxt('cs_c.txt', dtype=float, delimiter=', ', skip_header=1).T
# reads extraction rates of water
qwater = np.genfromtxt('cs_q.txt', dtype=float,
                       delimiter=', ', skip_header=1).T

# interpolates injection rates for given times
fco2 = interp1d(qco2[0, :], qco2[1, :], kind='linear',
                fill_value=(0., 0.), bounds_error=False)
qco2_fit = fco2(ts)
# interpolates extraction rates for given times
fwater = interp1d(qwater[0, :], qwater[1, :], kind='linear',
                  fill_value=(0., 0.), bounds_error=False)
qwater_fit = fwater(ts)

# computes net mass flow rates = injection - extraction
qs = qwater_fit - qco2_fit

data = np.genfromtxt('cs_p.txt', delimiter=',', skip_header=1)
tpr = data[:, 0]
pr = data[:, 1]
data = np.genfromtxt('cs_cc.txt', delimiter=',', skip_header=1)
tco = data[:, 0]
con = data[:, 1]


f, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(tp, qs, 'b', label="mass flow")
# ax.plot(tp, pro, 'kx', label="extractions")
ax2 = ax.twinx()
ax2.plot(tpr, pr, 'r-.', label="pressure")
ax2.set_ylabel("Pressure [MPa]")
ax.set_xlabel("Time [Year]")
ax.set_ylabel("Mass flow rate  [kg/s]")
ax.set_title("effect of mass flow rate on pressure")
ax.legend()
plt.show()

f, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(tp, qs, 'b-', label="mass flowrate")
ax.plot(ti, inj, 'kx', label="injections")
ax2 = ax.twinx()
ax2.plot(tco, con, 'r-.', label="concentration")
ax.set_xlabel("Time [Year]")
ax.set_ylabel("Mass flow rate [kg/s]")
ax2.set_ylabel("Concentration [wt%]")
ax.set_title("effect of mass flow rate on concentration")
ax.legend()
plt.show()
