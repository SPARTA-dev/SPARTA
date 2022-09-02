#                   ----------------------------------------
#                     run_PPDC_tests.py (SPARTA file)
#                   ----------------------------------------
# This file contains a few functions, used in the example notebooks to demonstrate the partial PDC periodograms.
#
# This file stores the following methods:
# ---------------------------------------------
#
# 1) simulate_planet_around_active_star - simulates observations of an active star,
#                                          either randomly or periodically active, orbited by a planets.
# 2) run_ppdc_tests - enables running the simulate_planet_around_active_star on a range of parameter values,
#                     for testing purposes.
#
# Dependencies: numpy, random, scipy, matplotlib and copy.
# Last update: Avraham Binnenfeld, 20210510.

from sparta.UNICOR.Spectrum import Spectrum
from sparta.UNICOR.Template import Template
from sparta.Auxil.TimeSeries import TimeSeries
from sparta.Observations import Observations
import numpy as np
import random
from scipy import interpolate
import matplotlib.pyplot as plt
from copy import deepcopy
from examples.run_USuRPER_tests import simulate_kepler_ellipse


def simulate_planet_around_active_star(v_sin_i, epsilon, integration_ratio, star_template, template_spot,
                                       p_spot, p_planet, spec_power_ratio, planet_k, star_k, planet_param,
                                       N, snr, periocic_spot_flag, seed=-1):

    new_temp = Template(template=Spectrum(wv=star_template.model.wv,
                                          sp=star_template.model.sp).InterpolateSpectrum(delta=0.5))

    new_temp.RotationalBroadening(epsilon=epsilon, vsini=v_sin_i)

    template_star_broadend = deepcopy(Template(template=Spectrum(wv=[new_temp.model.wv[0][60:-60]],
                                                                 sp=[new_temp.model.sp[0][60:-60]]).SpecPreProccess()))
    if seed != -1:
        random.seed(seed)

    times = [(random.random() * 100) for _ in range(N)]

    if periocic_spot_flag:
        vals_spot = [star_k * np.sin(2 * t * np.pi / p_spot) for t in times]
        for i, t in enumerate(times):
            if abs(t - p_spot/2) < 10:
                vals_spot[i] = 0

    else:
        std_spot_i = 0.5
        mu, sigma = 0, std_spot_i  # mean and standard deviation
        vals_spot = [star_k * np.random.normal(mu, sigma) for _ in times]

    keplerian_velocities = simulate_kepler_ellipse(times, planet_param, p_planet)
    vals_planet = [planet_k * v for v in keplerian_velocities]

    visit_spec_list = []

    for i, v in enumerate(vals_spot):
        new_wl_spot = star_template.doppler(v)

        z1 = interpolate.interp1d(new_wl_spot[0], star_template.model.sp[0], kind='quadratic')
        z2 = interpolate.interp1d(template_star_broadend.model.wv[0], template_star_broadend.model.sp[0], kind='quadratic')

        spotted_spec = z1(template_star_broadend.model.wv[0][60:-60]) * - spec_power_ratio + template_star_broadend.model.sp[0][60:-60]

        spotted_wl = template_star_broadend.model.wv[0][60:-60]

        spotted_t = Template(spectrum=spotted_spec, wavelengths=spotted_wl)

        new_wl_spot = template_spot.doppler(v)

        z1 = interpolate.interp1d(new_wl_spot[0], template_spot.model.sp[0], kind='quadratic')
        z2 = interpolate.interp1d(spotted_t.model.wv[0], spotted_t.model.sp[0], kind='quadratic')

        spotted_spec = z1(spotted_t.model.wv[0][60:-60]) * spec_power_ratio + spotted_t.model.sp[0][60:-60]

        spotted_wl = spotted_t.model.wv[0][60:-60]

        spotted_t = Template(spectrum=spotted_spec, wavelengths=spotted_wl)

        if vals_planet[i] != 0:
            spotted_t_vel = spotted_t.doppler(vals_planet[i])
        else:
            spotted_t_vel = spotted_t.model.wv

        new_temp = Spectrum(wv=[spotted_t_vel[0]], sp=[spotted_t.model.sp[0]]).InterpolateSpectrum(delta=1)

        rot_flux = Template().GaussianBroadening(wv=new_temp.wv, sp=new_temp.sp, resolution=100_000)

        new_temp.sp = rot_flux

        if integration_ratio:
            wv, sp = Template().integrate_spec(integration_ratio=integration_ratio, wv=new_temp.wv, sp=new_temp.sp)
            new_temp.wv = wv
            new_temp.sp = sp

        if seed != -1:
            new_temp.sp = Template().add_noise(snr, new_temp.sp, rndseed=seed)
        else:
            new_temp.sp = Template().add_noise(snr, new_temp.sp)

        new_temp = new_temp.SpecPreProccess()

        visit_spec_list.append(new_temp)

    template_for_calc = Template(template=Spectrum(wv=[star_template.model.wv[0][60:-60]],
                                                                 sp=[star_template.model.sp[0][60:-60]]).SpecPreProccess())

    return N, times, visit_spec_list, template_for_calc # star_template # template_star_broadend


def run_ppdc_tests(N, v_sin_i, spec_power_ratio, planet_k, snr, template_star, template_spot, period, periocic_spot_flag):

    print("Details:", "N:", N, "v_sin_i:", v_sin_i, "spec_power_ratio:", spec_power_ratio,
          "planet_k:", planet_k, "snr:", snr, "...")

    N, times, visit_spec_list, template_star_broadend =\
        simulate_planet_around_active_star(v_sin_i=v_sin_i,
                                           epsilon=0.5,
                                           integration_ratio=[],
                                           star_template=template_star,
                                           template_spot=template_spot,
                                           p_spot=19,
                                           p_planet=7,
                                           spec_power_ratio=spec_power_ratio,
                                           star_k=1,
                                           planet_k=planet_k,
                                           planet_param=[],
                                           N=N,
                                           snr=snr,
                                           periocic_spot_flag=periocic_spot_flag)

    ts = TimeSeries(size=N, times=times, vals=visit_spec_list,
                    calculated_vrad_list=[])
    obs = Observations(time_series=ts)
    obs.initialize_periodicity_detector(freq_range=(1 / 1000, 0.5), periodogram_grid_resolution=1000)

    www = obs.calc_rv_against_template(template_star_broadend, dv=0.01,
                                 VelBound=[-015.5, 015.5], fastccf=True)

    calculated_vrad_list = www.vels

    _ = obs.ccf_list[3].plotCCFs()
    plt.show()

    calculated_ccf_peaks = obs.ccf_peaks

    obs.time_series.calculated_vrad_list = calculated_vrad_list
    obs.time_series.calculated_ccf_peaks = calculated_ccf_peaks

    obs.periodicity_detector.run_PDC_process(calc_biased_flag=False, calc_unbiased_flag=True)
    obs.periodicity_detector.run_USURPER_process(calc_biased_flag=False, calc_unbiased_flag=True)
    obs.periodicity_detector.run_Partial_periodogram_process(partial_type="shape")
    obs.periodicity_detector.run_Partial_periodogram_process(partial_type="shift")

    obs.periodicity_detector.run_GLS_process()

    obs.periodicity_detector.period_truth = [7, 19]
    obs.periodicity_detector.periodogram_plots(velocities_flag=True)

    plt.show()


if __name__ == '__main__':
    pass
