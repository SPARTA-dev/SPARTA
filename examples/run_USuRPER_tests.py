#                   ----------------------------------------
#                     USuRPer_testing_2020.py (SPARTA file)
#                   ----------------------------------------
# This file is simulating SB1, SB2 and Cepheid spectra,
# then running USuRPer and GLS on them and storing the results in the res folder.
#
# This file stores the following methods:
# ---------------------------------------------
#
# 1) simulate_kepler_ellipse - Generates a radial velocities list,
#                              representing the velocities for the specific described Keplerian orbit.
# 2) simulate_target - Generates astronomical target simulations, based on PHOENIX synthetic spectra.
# 3) run_tests - Executes the USuRPer checks comprised of simulating various target observations,
#                creating their periodograms and saving the results.
#
# Dependencies: numpy, random, scipy and PyAstronomy.
# Last update: Avraham Binnenfeld, 20200607.


from sparta.Auxil.PeriodicityDetector import PeriodicityDetector
from sparta.Auxil.TimeSeries import TimeSeries
from sparta.Observations import Observations
from sparta.UNICOR.Spectrum import Spectrum
from sparta.UNICOR.Template import Template
from sparta.UNICOR.CCF1d import CCF1d
import numpy as np
import random
import os
import sys
from scipy import signal
from scipy import interpolate
from PyAstronomy import pyasl
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# =============================================================================
def save_time_series(time_series, amp, snr, system_type):

    if not os.path.exists(r'res'):
        os.makedirs(r'res')

    details = [str(time_series.size), str(amp), str(snr), system_type]

    val_dict = {}

    for i in range(time_series.size):
        t_i = "time_" + str(i)
        s_i = "s_" + str(i)
        w_i = "w_" + str(i)

        val_dict[t_i] = pd.Series(time_series.times[i])
        val_dict[s_i] = pd.Series(time_series.vals[i].sp[0])
        val_dict[w_i] = pd.Series(time_series.vals[i].wv[0])

    output_data = pd.DataFrame.from_dict(val_dict)
    output_data.to_csv('res/' + "_".join(details) + '_ts_data' + '.csv')

    if time_series.calculated_vrad_list != []:
        if system_type == "sb1":
            output_data = pd.DataFrame.from_dict({"vrad": time_series.calculated_vrad_list,
                                                  "time": time_series.times},
                                                 orient="index")
            output_data.to_csv('res/' + "_".join(details) + '_vrad_data' + '.csv')

        elif system_type == "sb2":
            output_data = pd.DataFrame.from_dict({"vrad_1": time_series.calculated_vrad_list[0],
                                                  "vrad_2": time_series.calculated_vrad_list[1],
                                                  "time": time_series.times},
                                                 orient="index")
            output_data.to_csv('res/' + "_".join(details) + '_vrad_data' + '.csv')

        elif system_type == "cepheid":
            output_data = pd.DataFrame.from_dict({"temp": [5500 + temp for temp in time_series.calculated_vrad_list],
                                                  "time": time_series.times},
                                                 orient="index")
            output_data.to_csv('res/' + "_".join(details) + '_temp_data' + '.csv')

# =============================================================================
# =============================================================================
def save_res(observations, amp, snr, system_type, spec_list, p, additional_data=[], format=".png"):
    '''
    This function saves the results of the calculation into a "res" directory

    :param amp: half amplitude of the time series change
    :param snr: the signal to noise ratio of the time series
    :param system_type: the astronomical system the time series represents
    :param spec_list: list of spectrum objects needed for documentation
    :param p: the signal to noise ratio of the time series
    '''
    if not os.path.exists(r'res'):
        os.makedirs(r'res')

    details = [str(observations.observation_TimeSeries.size), str(amp), str(snr), system_type]

    observations.periodicity_detector.periodogram_plots()

    plt.suptitle("system_type: {}, half amp: {}, snr: {}, set size: {}".format(
        system_type, amp, snr, observations.observation_TimeSeries.size), y=1.00) # , size="medium"
    plt.savefig("res/" + "_".join(details) + '_plot' + format, dpi=100)

    output_data = pd.DataFrame.from_dict({"trial_freqs": observations.periodicity_detector.pdc_res_freqs,
                                          "power_unbiased": observations.periodicity_detector.pdc_res_power_unbiased,
                                          "power_biased": observations.periodicity_detector.pdc_res_power_biased,
                                          "gls:": observations.periodicity_detector.GLS_power},
                                         orient="index")
    output_data.to_csv('res/' + "_".join(details) + '_data' + '.csv')

    plt.close()

    if system_type == "sb1":
        times_phased = [t % p for t in observations.observation_TimeSeries.times]
        plt.scatter(times_phased, observations.observation_TimeSeries.calculated_vrad_list, marker='o', c='k')
        plt.title("system_type: {}, half amp: {}, snr: {}, set size: {}".format(
            system_type, amp, snr, observations.observation_TimeSeries.size))
        plt.xlabel("Time [day]")
        plt.ylabel("RV [km/s]")
        # plt.grid()
        plt.savefig('res/' + "_".join(details) + '_phase_folded_velocities' + format, dpi=100)
        plt.close()

        plt.scatter(observations.observation_TimeSeries.times, observations.observation_TimeSeries.calculated_vrad_list, marker='o', c='k')
        plt.title("system_type: {}, half amp: {}, snr: {}, set size: {}".format(
            system_type, amp, snr, observations.observation_TimeSeries.size))
        plt.xlabel("Time [day]")
        plt.ylabel("RV [km/s]")
        # plt.grid()
        plt.savefig('res/' + "_".join(details) + '_velocities' + format, dpi=100)
        plt.close()

        # ccf = CCF1d().CrossCorrelateSpec(template=Template(template=spec_list[0]),
        #                                 spec=spec_list[1]).CombineCCFs()
        # plt.plot(ccf.Corr['vel'], ccf.Corr['corr'][0])
        # plt.xlabel("RV [km / s]")
        # plt.title("system_type: {}, half amp: {}, snr: {}, set size: {}, resolution: {}.".format(
        #    system_type, amp, snr, observations.observation_TimeSeries.size, observations.periodicity_detector.periodogram_grid_resolution))
        # plt.savefig('res/' + "_".join(details) + '_ccf_example' + '.png', dpi=100)
        # plt.close()

    elif system_type == "sb2" and additional_data != []:
        times_phased = [t % p for t in observations.observation_TimeSeries.times]
        plt.scatter(times_phased, observations.observation_TimeSeries.calculated_vrad_list[0], marker='o', c='k')
        plt.scatter(times_phased, observations.observation_TimeSeries.calculated_vrad_list[1], marker='^', c='w', edgecolors='k')
        plt.legend(["Primary", "Secondary"])
        plt.xlabel("Time [day]")
        plt.ylabel("RV [km/s]")
        plt.title("system_type: {}, half amp: {}, snr: {}, set size: {}".format(
            system_type, amp, snr, observations.observation_TimeSeries.size))
        # plt.grid()
        plt.savefig('res/' + "_".join(details) + '_phase_folded_velocities' + format, dpi=100)
        plt.close()

        plt.scatter(observations.observation_TimeSeries.times, observations.observation_TimeSeries.calculated_vrad_list[0], marker='o', c='k')
        plt.scatter(observations.observation_TimeSeries.times, observations.observation_TimeSeries.calculated_vrad_list[1], marker='^', c='w', edgecolors='k')
        plt.legend(["Primary", "Secondary"])
        plt.xlabel("Time [day]")
        plt.ylabel("RV [km/s]")
        plt.title("system_type: {}, half amp: {}, snr: {}, set size: {}".format(
            system_type, amp, snr, observations.observation_TimeSeries.size))
        # plt.grid()
        plt.savefig('res/' + "_".join(details) + '_velocities' + format, dpi=100)
        plt.close()

        # ccf = additional_data[0]
        # plt.plot(ccf.Corr['vel'], ccf.Corr['corr'][0])
        # plt.title("system_type: {}, half amp: {}, snr: {}, set size: {}, resolution: {}.".format(
        #    system_type, amp, snr, observations.observation_TimeSeries.size, observations.periodicity_detector.periodogram_grid_resolution))
        # plt.xlabel("RV [km / s]")
        # plt.savefig('res/' + "_".join(details) + '_ccf_min' + '.png', dpi=100)
        # plt.close()

        # ccf = additional_data[1]
        # plt.plot(ccf.Corr['vel'], ccf.Corr['corr'][0])
        # plt.title("system_type: {}, half amp: {}, snr: {}, set size: {}, resolution: {}.".format(
        #     system_type, amp, snr, observations.observation_TimeSeries.size, observations.periodicity_detector.periodogram_grid_resolution))
        # plt.xlabel("RV [km / s]")
        # plt.savefig('res/' + "_".join(details) + '_ccf_max' + '.png', dpi=100)
        # plt.close()

    elif system_type == "cepheid":
        times_phased = [t % p for t in observations.observation_TimeSeries.times]
        temps = [temp + 5500 for temp in observations.observation_TimeSeries.calculated_vrad_list]
        plt.scatter(times_phased, temps, alpha=0.7, marker='o', c='k')
        plt.xlabel("Time [day]")
        plt.ylabel("Teff [K]")
        plt.title("system_type: {}, half amp: {}, snr: {}, set size: {}".format(
            system_type, amp, snr, observations.observation_TimeSeries.size))
        # plt.grid()
        plt.savefig('res/' + "_".join(details) + '_phased_folded_tempratures' + format, dpi=100)
        plt.close()

        plt.scatter(observations.observation_TimeSeries.times, temps, alpha=0.7, marker='o', c='k')
        plt.title("system_type: {}, half amp: {}, snr: {}, set size: {}".format(
            system_type, amp, snr, observations.observation_TimeSeries.size))
        plt.xlabel("Time [day]")
        plt.ylabel("Teff [K]")
        # plt.grid()
        plt.savefig('res/' + "_".join(details) + '_tempratures' + format, dpi=100)
        plt.close()

        # ccf = CCF1d().CrossCorrelateSpec(template=Template(template=spec_list[2]),
        #                                  spec=spec_list[3]).CombineCCFs()
        # plt.plot(ccf.Corr['vel'], ccf.Corr['corr'][0])

        # plt.title("system_type: {}, half amp: {}, snr: {}, set size: {}, resolution: {}.".format(
        #     system_type, amp, snr, observations.observation_TimeSeries.size, observations.periodicity_detector.periodogram_grid_resolution))
        # plt.xlabel("RV [km / s]")
        # plt.savefig('res/' + "_".join(details) + '_ccf_example' + '.png', dpi=100)
        # plt.close()



def simulate_kepler_ellipse(times):
    '''
    This function gets a list of times and generates a corresponding radial velocities list
    representing the velocities for the specific described Keplerian orbit.
    :param times: list of times for which the radial velocities will be calculated
    :return: list of radial velocities at the input times
    '''
    # Instantiate a Keplerian elliptical orbit with
    # semi-major axis of 1.3 length units,
    # a period of 2 time units, eccentricity of 0.3,
    # longitude of ascending node of 70 degrees, an inclination
    # of 10 deg, and a periapsis argument of 110 deg.
    ke = pyasl.KeplerEllipse(65, 7, e=0.3, Omega=70., i=10.0, w=110.0)

    # Get a time axis
    x = np.asanyarray(times)
    t = x

    # Calculate velocity on orbit
    vel = ke.xyzVel(t)

    vals = vel[::, 2]

    # normalize
    vals = [v / 10 for v in vals]

    return vals


def simulate_target(temp_spec, size, p, system_type, half_amp, snr, min_val, max_val, signal_type="sinus", faint_template=[], p2=[]):
    '''
    This function generates astronomical target simulations, based on PHOENIX synthetic spectra.
    :param temp_spec: PHOENIX spectrum the simulation will be based on
    :param size: desired simulated visit number.
    :param p: desired simulated period.
    :param system_type: astronomical system to be simulated; 'sb1', 'sb2' or 'cepheid'
    :param half_amp: simulated fluctuation wave half-amplitude
    :param snr: desired signal to noise ratio
    :param min_val: minimal value for wavelength range (Angstrom)
    :param max_val: maximal value for wavelength range (Angstrom)
    :param signal_type: simulated fluctuation wave shape (either "sinus", "KeplerEllipse" or "sawtooth")
    :param faint_template: PHOENIX secondary spectrum the simulation will be based on (for sb2 simulation)
    :return:
    '''

    random.seed(997)
    times = [(random.random() * 100) for _ in range(size)]
    if signal_type == "sinus":
        vals = [half_amp * np.sin(2 * t * np.pi / p) for t in times]
    elif signal_type == "sawtooth":
        vals = [half_amp * signal.sawtooth(2 * np.pi * 1 / p * t) for t in times]
    elif signal_type == "KeplerEllipse":
        keplerian_velocities = simulate_kepler_ellipse(times)
        vals = [half_amp * v for v in keplerian_velocities]
    else:
        raise ValueError('unknown signal_type.')

    visit_spec_list = []
    calculated_vrad_list = []

    original_template = Spectrum(wv=temp_spec.model.wv, sp=temp_spec.model.sp).SpecPreProccess()

    # adding the template as-is: useful for debugging
    # visit_spec_list.append(original_template)

    if system_type == "sb1":
        for i, v in enumerate(vals):
            new_wl = temp_spec.doppler(v)

            new_temp = Spectrum(wv=new_wl, sp=temp_spec.add_noise(snr)).SpecPreProccess()
            visit_spec_list.append(new_temp)

    elif system_type == "sb2":
        if signal_type == "sinus":
            vals_1 = [half_amp * np.sin(np.pi + 2 * t * np.pi / p) for t in times]
        elif signal_type == "KeplerEllipse":
            vals_1 = [-i for i in vals]

        faint_star = faint_template

        mass_ratio = temp_spec.PHXMASS / faint_star.PHXMASS

        for i, v in enumerate(vals):

            v = v * mass_ratio
            vals_1[i] = vals_1[i] * (1 / mass_ratio)

            new_wl_1 = temp_spec.doppler(vals_1[i])
            new_wl_2 = temp_spec.doppler(v)

            z1 = interpolate.interp1d(new_wl_1[0], temp_spec.model.sp, kind='quadratic') #
            z2 = interpolate.interp1d(new_wl_2[0], faint_star.model.sp, kind='quadratic') #

            ratio = (temp_spec.PHXREFF ** 2) / (faint_star.PHXREFF ** 2)
            sb2_spec = z1(new_wl_2[0][100:-100]) * ratio + z2(new_wl_2[0][100:-100])

            sb2_wl = new_wl_2[0][100:-100]

            lib_t = Template(spectrum=sb2_spec[0], wavelengths=sb2_wl, min_val=min_val, max_val=max_val,)

            new_temp = Spectrum(wv=lib_t.model.wv, sp=lib_t.add_noise(snr)).SpecPreProccess()

            temp_for_ccf = Spectrum(wv=[temp_spec.model.wv[0]], sp=temp_spec.model.sp).SpecPreProccess()

            visit_spec_list.append(new_temp)

        min_diff = 9999999
        max_diff = -999999

        max_i = 0
        min_i = 0

        for i in range(size):
            diff = abs(vals[i] % p - vals_1[i] % p)
            if diff > max_diff:
                max_diff = diff
                max_i = i
            if diff < min_diff:
                min_diff = diff
                min_i = i

        min_ccf = CCF1d().CrossCorrelateSpec(template=Template(template=temp_for_ccf), spec=visit_spec_list[min_i])
        max_ccf = CCF1d().CrossCorrelateSpec(template=Template(template=temp_for_ccf), spec=visit_spec_list[max_i])

        calculated_vrad_list = [min_ccf, max_ccf]

        vals = [vals_1, vals]

    elif system_type == "cepheid":

        temp = 5500
        log_g = 4.5
        metal = 0
        alpha = 0

        min_val = 4900
        max_val = 5100

        original_vals = vals
        vals_sub_100_vals = [int(v) % 100 for v in vals]
        vals = [int(v) - int(v) % 100 for v in vals]

        for i, v in enumerate(vals):

            lib_t_1 = Template(temp=temp + v, log_g=log_g, metal=metal, alpha=alpha,
                                       min_val=min_val, max_val=max_val, download=False)
            lib_t_2 = Template(temp=temp + v + 100, log_g=log_g, metal=metal, alpha=alpha,
                                       min_val=min_val, max_val=max_val, download=False)

            new_spec = (lib_t_1.model.sp[0] * (100 - vals_sub_100_vals[i])
                        + lib_t_2.model.sp[0] * (vals_sub_100_vals[i])) / 100

            lib_t = Template(spectrum=new_spec, wavelengths=lib_t_1.model.wv[0],
                                     min_val=min_val, max_val=max_val,)

            new_temp = Spectrum(wv=lib_t.model.wv, sp=lib_t.add_noise(snr)).SpecPreProccess()

            visit_spec_list.append(new_temp)

        calculated_vrad_list = []
        vals = original_vals

    elif system_type == "cepheid_sb1":

        temp = 5500
        log_g = 4.5
        metal = 0
        alpha = 0

        min_val = 4900
        max_val = 5100

        original_vals = vals
        vals_sub_100_vals = [int(v) % 100 for v in vals]
        vals = [int(v) - int(v) % 100 for v in vals]

        vals_sb1 = [10 * np.sin(2 * t * np.pi / p2) for t in times]

        for i, v in enumerate(vals):

            lib_t_1 = Template(temp=temp + v, log_g=log_g, metal=metal, alpha=alpha,
                                       min_val=min_val, max_val=max_val, download=False)
            lib_t_2 = Template(temp=temp + v + 100, log_g=log_g, metal=metal, alpha=alpha,
                                       min_val=min_val, max_val=max_val, download=False)

            new_spec = (lib_t_1.model.sp[0] * (100 - vals_sub_100_vals[i])
                        + lib_t_2.model.sp[0] * (vals_sub_100_vals[i])) / 100

            lib_t = Template(spectrum=new_spec, wavelengths=lib_t_1.model.wv[0],
                                     min_val=min_val, max_val=max_val,)

            lib_t.model.wv[0] = Template().doppler(vals_sb1[i], lib_t.model.wv[0])

            lib_t.model.wv = np.reshape(lib_t.model.wv[0], (1, -1))

            new_temp = Spectrum(wv=lib_t.model.wv, sp=lib_t.add_noise(snr)).SpecPreProccess()

            visit_spec_list.append(new_temp)

        calculated_vrad_list = []
        vals = original_vals

    return times, vals, calculated_vrad_list, visit_spec_list


def run_tests(system_type, size_list, noise_list, half_amp_list):
    '''
    This function executes the USuRPer checks comprised of simulating various target observations,
    creating their periodograms and saving the results.
    :param system_type: astronomical system to be simulated; 'sb1', 'sb2' or 'cepheid'
    :param size_list: list of sample sizes
    :param noise_list: list of snr values
    :param half_amp_list: list of half-amplitudes
    '''
    # Assigning sun-like stellar parameters to the simulated spectra
    temp = 5000 # 4900 88888888888 TBD
    log_g = 4.5
    metal = 0
    alpha = 0

    # Choosing wavelength range (Angstrom units)
    min_val = 4900
    max_val = 5100

    # Loading a Phoenix synthetic spectrum
    template = Template(temp=temp, log_g=log_g, metal=metal, alpha=alpha, min_val=min_val, max_val=max_val)

    # Generating USuRPer tests according to input parameters
    for size in size_list:
        for noise in noise_list:
            for half_amp in half_amp_list:

                print("Details:", system_type, "N:", size, "SNR:", noise, "q:", half_amp, "...")

                if system_type == "sb1":

                    print("Generating SB1 simulated spectra. ", end = '', flush=True)

                    # Disable output
                    sys.stdout = open(os.devnull, 'w')
                    visit_time_list, visit_vrad_list, calculated_vrad_list, spec_list =\
                        simulate_target(template, size=size, p=7, system_type=system_type, half_amp=half_amp, snr=noise,
                                        min_val=min_val, max_val=max_val,)# signal_type="KeplerEllipse"
                    # Enable output:
                    sys.stdout = sys.__stdout__

                    print("Assigning to TimeSeries. ", flush=True)
                    ts = TimeSeries(size=size, times=visit_time_list, vals=spec_list,
                                           calculated_vrad_list=calculated_vrad_list)
                    obs = Observations(time_series=ts)
                    obs.initialize_periodicity_detector(freq_range=(1/1000, 1), periodogram_grid_resolution=1000)
                    print("Starting USuRPER... ", end='', flush=True)
                    # Disable output
                    sys.stdout = open(os.devnull, 'w')
                    calculated_vrad_list = obs.calc_rv_against_template(template).vels
                    obs.observation_TimeSeries.calculated_vrad_list = calculated_vrad_list
                    obs.periodicity_detector.run_USURPER_process(calc_biased_flag=False, calc_unbiased_flag=True)
                    # Enable output:
                    sys.stdout = sys.__stdout__

                    print("Done.\nCalculating GLS... ", end='', flush=True)
                    obs.periodicity_detector.run_GLS_process()
                    print("Done.")

                elif system_type == "sb2":

                    print("Generating SB2 simulated spectra. ", end = '', flush=True)
                    # Disable output
                    sys.stdout = open(os.devnull, 'w')

                    faint_temp = Template(temp=5500, log_g=log_g, metal=metal, alpha=alpha, min_val=min_val,
                                                  max_val=max_val, download=False) # 4000

                    visit_time_list, visit_vrad_list, calculated_vrad_list, spec_list =\
                        simulate_target(template, size=size, p=7, system_type=system_type, half_amp=half_amp, snr=noise,
                                        min_val=min_val, max_val=max_val, faint_template=faint_temp, signal_type="KeplerEllipse")
                    # Enable output:
                    sys.stdout = sys.__stdout__

                    print("Assigning to TimeSeries. ", flush=True)
                    ts = TimeSeries(size=size, times=visit_time_list, vals=spec_list,
                                           calculated_vrad_list=visit_vrad_list)
                    obs = Observations(time_series=ts)
                    obs.initialize_periodicity_detector(freq_range=(1/1000, 1), periodogram_grid_resolution=1000)
                    print("Starting USuRPER... ", end='', flush=True)
                    # Disable output
                    sys.stdout = open(os.devnull, 'w')
                    obs.periodicity_detector.run_USURPER_process(calc_biased_flag=False, calc_unbiased_flag=True)
                    # Enable output:
                    sys.stdout = sys.__stdout__
                    print("Done.", flush=True)

                elif system_type == "cepheid":

                    print("Generating Cepheid-like spectra. ", end = '', flush=True)
                    # Disable output
                    sys.stdout = open(os.devnull, 'w')

                    visit_time_list, visit_vrad_list, calculated_vrad_list, spec_list =\
                        simulate_target(template, size=size, p=7, system_type=system_type, half_amp=half_amp, snr=noise,
                                        min_val=min_val, max_val=max_val, signal_type="sawtooth")
                    # Enable output:
                    sys.stdout = sys.__stdout__

                    print("Assigning to TimeSeries. ", flush=True)
                    ts = TimeSeries(size=size, times=visit_time_list, vals=spec_list,
                                           calculated_vrad_list=visit_vrad_list)
                    obs = Observations(time_series=ts)
                    obs.initialize_periodicity_detector(freq_range=(1/1000, 1), periodogram_grid_resolution=1000)

                    print("Starting USuRPER... ", end='', flush=True)
                    # Disable output
                    sys.stdout = open(os.devnull, 'w')
                    obs.periodicity_detector.run_USURPER_process(calc_biased_flag=False, calc_unbiased_flag=True)
                    # Enable output:
                    sys.stdout = sys.__stdout__
                    print("Done.", flush=True)

                elif system_type == "cepheid_sb1":

                    print("Generating Cepheid-like SB1 spectra. ", end = '', flush=True)
                    # Disable output
                    sys.stdout = open(os.devnull, 'w')

                    visit_time_list, visit_vrad_list, calculated_vrad_list, spec_list =\
                        simulate_target(template, size=size, p=5, system_type=system_type, half_amp=half_amp, snr=noise,
                                        min_val=min_val, max_val=max_val, signal_type="sawtooth", p2=3)
                    # Enable output:
                    sys.stdout = sys.__stdout__

                    print("Assigning to TimeSeries. ", flush=True)
                    ts = TimeSeries(size=size, times=visit_time_list, vals=spec_list,
                                           calculated_vrad_list=visit_vrad_list)
                    obs = Observations(time_series=ts)
                    obs.initialize_periodicity_detector(freq_range=(1/1000, 1), periodogram_grid_resolution=1000)

                    print("Starting USuRPER... ", end='', flush=True)
                    # Disable output
                    sys.stdout = open(os.devnull, 'w')
                    obs.periodicity_detector.run_USURPER_process(calc_biased_flag=False, calc_unbiased_flag=True)
                    # Enable output:
                    sys.stdout = sys.__stdout__
                    print("Done.", flush=True)


                save_time_series(time_series=ts, amp=half_amp, snr=noise, system_type=system_type)
                save_res(observations=obs, amp=half_amp, snr=noise, system_type=system_type, spec_list=spec_list, p=7,
                            additional_data=calculated_vrad_list)



                print("Simulation completed.\n\n")

if __name__ == '__main__':
    '''
    An example code showing some of the runs executed testing the USuRPer.
    '''

    print("""\n\n
    USuRPER -- Unit Sphere Representation PERiodogram
    --------------------------------------------------

    This routine generates three simulations: SB1, SB2, Cepheid and Cepheid sb1.
    In each simulation, spectra will be generated and analyzed.
    The simulated modulation will have periodicity of seven days. 
    Summary plots will be saved in the ./res folder. 

    Starting simulation. This will take ~50 minutes.\n\n
    """)

    run_tests(system_type="sb1", size_list=[50], half_amp_list=[10], noise_list=[100])

    run_tests(system_type="sb2", size_list=[20], half_amp_list=[10], noise_list=[30])

    run_tests(system_type="cepheid", size_list=[15], half_amp_list=[500], noise_list=[30])

    run_tests(system_type="cepheid_sb1", size_list=[50], half_amp_list=[500], noise_list=[100])

