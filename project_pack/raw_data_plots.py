from matplotlib.markers import MarkerStyle
import numpy as np
from matplotlib import pyplot as plt
from pressure_model import fetch_pressure_data
from concentration_model import fetch_concentration_data

def plot_raw_data():
    ''' Plots pressure and concentration measurements, and the CO2 injection and mass extraction measurements.
    
        Parameters
        ----------
        None

        Returns
        -------
        None
    '''
    # read pressure data 
    ts_P,Ps = fetch_pressure_data()
    # read concentration data
    ts_C,Cs = fetch_concentration_data()
    
    # read injection data 
    inj_data = np.genfromtxt('cs_c.txt',dtype=float,delimiter=', ',skip_header=1).T
    # seperate time and injection data lists
    ts_inj = inj_data[0,:]
    injs = inj_data[1,:]

    # read extraction data 
    ext_data = np.genfromtxt('cs_q.txt',dtype=float,delimiter=', ',skip_header=1).T
    # seperate time and extraction data lists
    ts_ext = ext_data[0,:]
    exts = ext_data[1,:]

    # create two side-by-side plots
    f,ax = plt.subplots(1,2)

    # plots pressures on left side
    ax[0].plot(ts_P,Ps,linestyle='-',marker='o',color='b',label='Pressure')
    # creates twin axis for concentration
    ax2 = ax[0].twinx()
    # plots concentration on left side
    ax2.plot(ts_C,Cs,linestyle='-',marker='o',color='r',label='Concentration')

    # plots injection rate on right side
    ax[1].plot(ts_inj,injs,linestyle='-',marker='o',color='m',label='Injection rate')
    # creates twin axis for extraction
    ax3 = ax[1].twinx()
    # plots extraction on right side
    ax3.plot(ts_ext,exts,linestyle='-',marker='o',color='y',label='Extraction rate')

    # adds title to both plots
    ax[0].set_title('Measured pressures and CO2 concentrations\n in the Ohaaki reservoir')
    ax[1].set_title('CO2 injection rates and mass extraction rates\n in the Ohaaki reservoir')

    # sets axis labels
    ax[0].set_xlabel('Year [A.D.]')
    ax[1].set_xlabel('Year [A.D.]')
    ax[0].set_ylabel('Pressure [MPa]')
    ax2.set_ylabel('Concentration of CO2 [wt.%]')
    ax[1].set_ylabel('CO2 injection rate [kg/s]')
    ax3.set_ylabel('Mass extraction rate [kg/s]')

    # adds legends
    ax2.legend(loc='upper right')
    ax[0].legend(loc='upper left')
    ax[1].legend(loc='upper right')
    ax3.legend(loc='upper left')

    # adjusts padding between plots
    f.tight_layout(pad=10.)
    # set font size for axis labels and title
    plt.rcParams.update({'font.size': 14})
    # shows plot
    plt.show()

def main():
    plot_raw_data()

if __name__ == '__main__':
    main()