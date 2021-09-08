import pressure_model
import pressure_calibration
import pressure_predictions
import concentration_model
import concentration_calibration
import concentration_predictions

def main():
    # initial guess of parameters
    pars0 = [5.e+03, 2.5e-3,  3.e-01, 8.e-04, 5.e-01,  6.17e+00]

    # plot initial pressure model
    pressure_model.plot_pressure(pars0)
    pressure_calibration.plot_pressure_residuals(pars0)
    # plot initial concentration model
    concentration_model.plot_concentration(pars0)
    concentration_calibration.plot_concentration_residuals(pars0)

    # calibrate model to fit best fit parameters
    pars = concentration_calibration.calibrate_concentration_model(pars0)
    # plot best fit pressure model
    pressure_model.plot_pressure(pars)
    pressure_calibration.plot_pressure_residuals(pars)
    # plot best fit concentration model
    concentration_model.plot_concentration(pars)
    concentration_calibration.plot_concentration_residuals(pars)

    # plot what-ifs of pressure model
    pressure_predictions.plot_predictions(pars)
    # plot what-ifs of concentration model
    concentration_predictions.plot_predictions(pars)


if __name__ == "__main__":
    main()