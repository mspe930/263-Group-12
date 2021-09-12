import pressure_model
import pressure_calibration
import pressure_predictions
import pressure_uncertainty
import concentration_model
import concentration_calibration
import concentration_predictions
import concentration_uncertainty
import raw_data_plots
import instability
import analytic_soln
import raw_data

def main():
    # plot raw data
    raw_data_plots.plot_raw_data()
    raw_data.main()

    # initial guess of parameters
    pars0 = [5.e+03, 2.5e-3,  3.e-01, 8.e-04, 5.e-01,  6.17e+00]
    # calibrated parameters
    pars = concentration_calibration.calibrate_concentration_model(pars0)

    # plot benchmarked pressure model against numerical soln
    analytic_soln.plot_pressure_benchmark(*pars0)
    # plot benchmarked concentration model against numerical soln
    analytic_soln.plot_concentration_benchmark(*pars0)
    # plot pressure numerical soln at large time step
    instability.plot_pressure_time_step(*pars0, t_step=10)
    # plot concentraion numerical soln at large time step
    instability.plot_concentration_time_step(*pars0, t_step=10)

    # initially assume that C' = C(t) always
    pressure_model.plot_pressure(pars)  # pressure plot
    pressure_calibration.plot_pressure_residuals(pars) # pressure residual plot
    concentration_model.plot_concentration(pars,False)    # concentration plot
    concentration_calibration.plot_concentration_residuals(pars,False) # concentration residual plot

    # now assume that C' takes on a piecewise form given in ODE
    concentration_model.plot_concentration(pars,True)    # concentration plot
    concentration_calibration.plot_concentration_residuals(pars,True) # concentration residual plot


    # plot what-ifs of pressure model
    pressure_predictions.plot_predictions(pars)
    # plot what-ifs of concentration model
    concentration_predictions.plot_predictions(pars)

    # plot what-ifs with uncertainty of pressure model
    pressure_uncertainty.plot_pressure_posterior(pars)
    # plot what-ifs with uncertainty of concentration model
    concentration_uncertainty.plot_predictions_uncert(pars)

if __name__ == "__main__":
    main()
