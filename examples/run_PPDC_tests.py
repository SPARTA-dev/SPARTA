#                   ----------------------------------------
#                     run_PPDC_tests.py (SPARTA file)
#                   ----------------------------------------
# This file is simulating SB1, SB2 and Cepheid spectra,
# then running 8888 and storing the results in the res folder.
#
# This file stores the following methods:
# ---------------------------------------------
#
# 1) -
# 2) -
#
# Dependencies: .
# Last update: Avraham Binnenfeld, 20200404.


from sparta.Auxil.PeriodicityDetector import PeriodicityDetector
from sparta.UNICOR.Spectrum import Spectrum
from sparta.UNICOR.Template import Template
from sparta.Auxil.TimeSeries import TimeSeries
from sparta.Observations import Observations
import numpy as np
import random
from scipy import interpolate
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
from scipy import signal
from sparta.UNICOR.CCF1d import CCF1d
import winsound
from copy import deepcopy



def simulate_kepler_ellipse(times, params=[], p=7):
    '''
    This function gets a list of times and generates a corresponding radial velocities list
    representing the velocities for the specific described Keplerian orbit.
    :param times: list of times for which the radial velocities will be calculated
    :return: list of radial velocities at the input times
    '''
    # Instantiate a Keplerian elliptical orbit with
    # semi-major axis of 1.3 length units,
    # a period of 10 time units, eccentricity of 0.3,
    # longitude of ascending node of 70 degrees, an inclination
    # of 10 deg, and a periapsis argument of 110 deg.
    ke = pyasl.KeplerEllipse(65, p, e=0, Omega=70., i=10.0, w=110.0) # 0.3

    # Get a time axis
    x = np.asanyarray(times)
    t = x

    # Calculate velocity on orbit
    vel = ke.xyzVel(t)

    vals = vel[::, 2]

    # normalize
    vals = [v / 10 for v in vals]

    return vals


def simulate_planet_around_active_star(v_sin_i, epsilon, integration_ratio, star_template, template_spot, p_spot, p_planet,
                                       spec_power_ratio, planet_k, star_k, planet_param, N, snr, min_val, max_val):

    new_temp = Template(template=Spectrum(wv=star_template.model.wv, sp=star_template.model.sp).InterpolateSpectrum(delta=0.5)) # .SpecPreProccess())

    new_temp.RotationalBroadening(epsilon=epsilon, vsini=v_sin_i)

    template_star_broadend = deepcopy(Template(template=Spectrum(wv=[star_template.model.wv[0][60:-60]], sp=[star_template.model.sp[0][60:-60]]).SpecPreProccess()))

    # random.seed(995)
    times = [(random.random() * 100) for _ in range(N)]
    # times = [i for i in range(N)]

    std_spot_i = 0.5
    mu, sigma = 0, std_spot_i  # mean and standard deviation
    vals_spot = [1 * np.random.normal(mu, sigma) for _ in times]
    # vals_spot = [1 * np.sin(2 * t * np.pi / p_spot) for t in times] # star_k
    # vals_spot = [1 * random.uniform(-1, 1) for _ in times] # star_k


    keplerian_velocities = simulate_kepler_ellipse(times, planet_param, p_planet)
    vals_planet = [planet_k * v for v in keplerian_velocities]

    visit_spec_list = []

    for i, v in enumerate(vals_spot):
        new_wl_spot = template_spot.doppler(v)

        z1 = interpolate.interp1d(new_wl_spot[0], template_spot.model.sp[0], kind='quadratic')
        z2 = interpolate.interp1d(star_template.model.wv[0], star_template.model.sp[0], kind='quadratic')

        spotted_spec = z1(star_template.model.wv[0][60:-60]) * spec_power_ratio + star_template.model.sp[0][60:-60] #  z2(star_template.model.wv[0][60:-60])

        spotted_wl = star_template.model.wv[0][60:-60]

        spotted_t = Template(spectrum=spotted_spec, wavelengths=spotted_wl)

        if vals_planet[i] != 0:
            spotted_t_vel = spotted_t.doppler(vals_planet[i]) # spotted_t.model.wv#
        else:
            spotted_t_vel = spotted_t.model.wv

        new_temp = Spectrum(wv=[spotted_t_vel[0]], sp=[spotted_t.model.sp[0]]).InterpolateSpectrum(delta=1) # .SpecPreProccess()

        rot_flux = Template().GaussianBroadening(wv=new_temp.wv, sp=new_temp.sp, resolution=100_000)

        new_temp.sp = rot_flux

        if integration_ratio:
            wv, sp = Template().integrate_spec(integration_ratio=integration_ratio, wv=new_temp.wv, sp=new_temp.sp)
            new_temp.wv = wv
            new_temp.sp = sp
            pass

        new_temp.sp = Template().add_noise(snr, new_temp.sp)

        new_temp = new_temp.SpecPreProccess()

        visit_spec_list.append(new_temp)

    return N, times, visit_spec_list, template_star_broadend


def simulate_cepheid(k_temp, planet_k, p_ceph, planet_param, snr, N):

    # times = [i for i in range(N)]
    times = [(random.random() * 100) for _ in range(N)]
    vals_temp = [k_temp * np.sin(2 * t * np.pi / p_ceph) for t in times] # v_sin_i
    # vals_temp = [k_temp * signal.sawtooth(2 * np.pi * 1 / p_ceph * t) for t in times]

    keplerian_velocities = simulate_kepler_ellipse(times, params=planet_param)
    vals_planet = [planet_k * v for v in keplerian_velocities]

    visit_spec_list = []

    temp = 5500
    log_g = 4.5
    metal = 0
    alpha = 0

    min_val = 4900
    max_val = 5100

    vals = vals_temp

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

        # lib_t.RotationalBroadening(epsilon=0.5, vsini=2)

        new_temp = Spectrum(wv=lib_t.model.wv, sp=lib_t.add_noise(snr)).SpecPreProccess()

        new_temp = Template(template=new_temp)

        new_temp.RotationalBroadening(epsilon=0.5, vsini=2)

        new_temp.model.wv[0] = Template().doppler(vals_planet[i], new_temp.model.wv[0])

        new_temp.model.wv = np.reshape(new_temp.model.wv[0], (1, -1))

        new_temp = Spectrum(wv=new_temp.model.wv, sp=new_temp.model.sp).SpecPreProccess()

        visit_spec_list.append(new_temp)

        plt.plot(new_temp.wv[0], new_temp.sp[0])

    calculated_vrad_list = []
    vals = original_vals

    plt.show()

    return N, times, visit_spec_list, Template(temp=temp, log_g=log_g, metal=metal, alpha=alpha,
                           min_val=min_val, max_val=max_val, download=False)

def test_velocity_std(template_star, template_spot, epsilon, v_sin_i, n, std_spot=[], snr=[], spec_power_ratio=0.001):

    new_temp = Template(template=Spectrum(wv=template_star.model.wv, sp=template_star.model.sp).SpecPreProccess())
    new_temp.RotationalBroadening(epsilon=epsilon, vsini=v_sin_i)

    std_list = []
    for std_spot_i in std_spot:
        snr_vrad_list = []
        mu, sigma = 0, std_spot_i  # mean and standard deviation
        times = [(random.random() * 100) for _ in range(n)]
        vals_spot = [1 * np.random.normal(mu, sigma) for t in times]

        for snr_i in snr:
            for j in range(n):

                new_wl_spot = template_spot.doppler(vals_spot[j])

                z1 = interpolate.interp1d(new_wl_spot[0], template_spot.model.sp[0], kind='quadratic')
                z2 = interpolate.interp1d(template_star.model.wv[0], template_star.model.sp[0], kind='quadratic')

                spotted_spec = z1(new_temp.model.wv[0][60:-60]) * spec_power_ratio + new_temp.model.sp[0][60:-60]

                spotted_wl = new_temp.model.wv[0][60:-60]

                spotted_t = Template(spectrum=spotted_spec, wavelengths=spotted_wl)

                rot_flux = Template().GaussianBroadening(wv=spotted_t.model.wv, sp=spotted_t.model.sp, resolution=100_000)

                spotted_t.model.sp = rot_flux

                spec_j = Spectrum(wv=spotted_t.model.wv, sp=spotted_t.model.sp).SpecPreProccess()
                spec_j.sp = Template().add_noise(snr_i, spec_j.sp)

                ccf = CCF1d().CrossCorrelateSpec(template=new_temp, spec=spec_j, dv=0.05, VelBound=0.5)

                snr_vrad_list.append(ccf.Corr['RV'][0][0])
        std_list.append(np.std(snr_vrad_list))

        # plt.scatter([k for k in range(n)], snr_vrad_list)
        # plt.title()

    plt.grid()
    plt.scatter(np.log10(std_spot), np.log10([i * 1_000 for i in std_list]))
    plt.xlabel("log10 std_spot")
    plt.ylabel("log10 vrad std (m/s)")
    plt.show()

    pass

def run_ppdc_tests(N, v_sin_i, spec_power_ratio, planet_k, snr, template_star, template_spot, min_val, max_val):

    sde_planet_list = []
    sde_spot_list = []
    gls_sde_planet_list = []
    gls_sde_spot_list = []

    for n in N:
        for v in v_sin_i:
            for r in spec_power_ratio:
                for k_p in planet_k:
                    for noise in snr:

                        Details = "_".join(["N", str(n), "v_sin_i", str(v), "spec_power_ratio", str(r), "planet_k", str(k_p), "snr", str(noise)])

                        print("Details:", "N:", n, "v_sin_i:", v, "spec_power_ratio:", r, "planet_k:", k_p, "snr:", noise, "...")

                        # test_velocity_std(template_star=template_star, template_spot=template_spot, epsilon=0.5, v_sin_i=6, n=200, std_spot=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6], snr=[10])

                        N, times, visit_spec_list, template_star_broadend =\
                            simulate_planet_around_active_star(v_sin_i=v,
                                                               epsilon=0.5,
                                                               integration_ratio=[],
                                                               star_template=template_star,
                                                               template_spot=template_spot,
                                                               p_spot=19,
                                                               p_planet=7,
                                                               spec_power_ratio=r,
                                                               planet_k=k_p,
                                                               star_k=v,
                                                               planet_param=[],
                                                               N=n,
                                                               snr=noise,
                                                               min_val=min_val,
                                                               max_val=max_val)

                        ts = TimeSeries(size=N, times=times, vals=visit_spec_list,
                                        calculated_vrad_list=[])
                        obs = Observations(time_series=ts)
                        obs.initialize_periodicity_detector(freq_range=(1 / 1000, 0.5), periodogram_grid_resolution=1000)

                        calculated_vrad_list = obs.calc_rv_against_template(template_star_broadend, dv=0.01,
                                                     VelBound=[-0.5, 0.5]).vels

                        calculated_ccf_peaks = obs.ccf_peaks

                        # # 88888
                        # calculated_vrad_list = [0.005 * np.sin(2 * t * np.pi / 7) for t in times]
                        # calculated_ccf_peaks = [0.7 + 0.1 * np.sin(2 * t * np.pi / 3) for t in times]
                        # # calculated_ccf_peaks = [(0.6 + random.random() * 0.1) for t in times]
                        #
                        # spot_factor = 0.005
                        #
                        # for i, _ in enumerate(times):
                        #     calculated_vrad_list[i] = calculated_vrad_list[i] + spot_factor * calculated_ccf_peaks[i]

                        # NormFac = np.quantile(calculated_vrad_list, 0.98)
                        #
                        # noise = np.random.normal(0, 1, len(calculated_vrad_list)) * NormFac / 25
                        #
                        # calculated_vrad_list = calculated_vrad_list + noise

                        obs.observation_TimeSeries.calculated_vrad_list = calculated_vrad_list
                        obs.observation_TimeSeries.calculated_ccf_peaks = calculated_ccf_peaks

                        obs.periodicity_detector.run_PDC_process(calc_biased_flag=False, calc_unbiased_flag=True)
                        # obs.periodicity_detector.run_PDC_process(calc_biased_flag=True, calc_unbiased_flag=False)
                        obs.periodicity_detector.run_USURPER_process(calc_biased_flag=False, calc_unbiased_flag=True)
                        obs.periodicity_detector.run_Partial_USURPER_process(reversed_flag=True)
                        obs.periodicity_detector.run_Partial_USURPER_process(reversed_flag=False)

                        # obs.periodicity_detector.run_PDC_process(calc_biased_flag=False, calc_unbiased_flag=True)
                        # obs.periodicity_detector.run_USURPER_process(calc_biased_flag=False, calc_unbiased_flag=True)


                        obs.periodicity_detector.run_GLS_process()

                        obs.periodicity_detector.period = [7, 19] # 88888
                        obs.periodicity_detector.periodogram_plots(velocities_flag=True)

                        plt.savefig("res/" + Details + '_plot.png', dpi=100)

                        array = np.asarray(obs.periodicity_detector.pdc_res_power_unbiased)
                        p_planet_index = (np.abs(array - 1 / 7)).argmin()
                        p_spot_index = (np.abs(array - 1 / 19)).argmin()

                        std = np.std(array)

                        sde_planet = array[p_planet_index] / std
                        sde_spot = array[p_spot_index] / std

                        sde_planet_list.append(sde_planet)
                        sde_spot_list.append(sde_spot)

                        gls_sde_planet_list = []
                        gls_sde_spot_list = []

                        plt.close()

                        frequency = 2500  # Set Frequency To 2500 Hertz
                        duration = 500  # Set Duration To 1000 ms == 1 second
                        winsound.Beep(frequency, duration)

                        # plt.scatter(sde_planet_list, sde_spot_list, alpha=0.3)
                        #
                        # plt.xlabel("sde_planet")
                        # plt.ylabel("sde_spot")
                        # plt.title("res SDE scatter plot")
                        #
                        # plt.savefig("res/" + Details + '_sde.png', dpi=100)


if __name__ == '__main__':
    '''
    An example code showing some of the runs executed testing the USuRPer.
    '''

    print("""\n\n
    ---
    --------------------------------------------------

    Starting simulation. .\n\n
    """)

    # Assigning sun-like stellar parameters to the simulated spectra
    temp_star = 5800
    temp_spot = 5800 # 2300

    log_g = 4.5
    metal = 0
    alpha = 0

    # Choosing wavelength range (Angstrom units)
    min_val = 5920 # 5500 5880
    max_val = 6000 # 6700 5920

    # Loading a Phoenix synthetic spectrum
    template_star = Template(temp=temp_star, log_g=log_g, metal=metal, alpha=alpha, min_val=min_val, max_val=max_val)
    template_spot = Template(temp=temp_spot, log_g=log_g, metal=metal, alpha=alpha, min_val=min_val, max_val=max_val)

    v_sin_i = [6]
    spec_power_ratio = [-0.020] # -0.005
    planet_k = [0.005] # 50!!!!!
    snr = [-1]
    N = [23, 24, 25, 26]

    run_ppdc_tests(N, v_sin_i, spec_power_ratio, planet_k, snr, template_star, template_spot, min_val, max_val)
