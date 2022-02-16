"""
from https://github.com/rpoleski/MulensModel/blob/master/examples/example_18_simulate.py
and https://github.com/LRayleighJ/MDN_lc_iden/blob/main/simudata/gen_simu.py

Script for simulating microlensing lightcurves.

NB: output mag = 22 - 2.5lg(flux)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

import MulensModel as mm

import random
import h5py
from tqdm import tqdm
import time 
from multiprocessing import Pool
from functools import partial


def simulate_lc(
        parameters, time_settings,
        coords=None, methods=None,
        flux_source=1000., flux_blending=0.,
        relative_uncertainty=0.01,
        plot=True):
    """
    Simulate and save light curve.

    Parameters :
        parameters: *dict*
            Parameters of the model - keys are in MulensModel format, e.g.,
            't_0', 'u_0', 't_E' etc.

        time_settings: *dict*
            Sets properties of time vector. It requires key `type`, which can
            have one of two values:
            - `random` (requires `n_epochs`, `t_start`, and `t_stop`) or
            - `evenly spaced` (settings passed to `Model.set_times()`).

        coords: *str*
            Event coordinates for parallax calculations, e.g.,
            "17:34:51.15 -30:29:28.27".

        methods: *list*
            Define methods used to calculate magnification. The format is
            the same as MulensModel.Model.set_magnification_methods().

        flux_source: *float*
            Flux of source.

        flux_blending: *float*
            Blending flux.

        relative_uncertainty: *float*
            Relative uncertainty of the simulated data (this is close to
            sigma in magnitudes).

        plot: *bool*
            Plot the data and model at the end?

    """
    try:
        model = mm.Model(parameters, coords=coords)

        if time_settings['type'] == 'random':
            raw = np.random.rand(time_settings['n_epochs'])
            dt = time_settings['t_stop'] - time_settings['t_start']
            times = time_settings['t_start'] + np.sort(raw) * dt
        elif time_settings['type'] == 'evenly spaced':
            times = model.set_times(t_start=time_settings['t_start'],
                                        t_stop=time_settings['t_stop'],
                                        n_epochs=time_settings['n_epochs'])
        else:
            raise ValueError("unrecognized time_settings['type']: " +
                            time_settings['type'])

        if methods is not None:
            model.set_magnification_methods(methods)

        magnification = model.get_magnification(times)

        flux = flux_source * magnification + flux_blending
        flux_err = relative_uncertainty * flux

        flux *= 1 + relative_uncertainty * np.random.randn(len(flux))

        data = mm.MulensData([times, flux, flux_err], phot_fmt='flux')
        # event = mm.Event([data], model)
        # print("chi^2: {:.2f}".format(event.get_chi2()))

        # np.savetxt(file_out,
        #            np.array([times, data.mag, data.err_mag]).T,
        #            fmt='%.4f')

        if plot:
            model.plot_lc(t_start=np.min(times), t_stop=np.max(times),
                        source_flux=flux_source, blend_flux=flux_blending)
            data.plot(phot_fmt='mag')
            plt.savefig('./test.png')

        if time_settings['type'] == 'random':
            single = mm.Model({'t_0': parameters['t_0'], 'u_0': parameters['u_0'], 't_E': parameters['t_E']})
            event_single = mm.Event([data], single)
            chi2 = event_single.get_chi2()
            # print("chi^2 single: {:.2f}".format(chi2))

            if chi2 > 72./0.01:
                # plt.plot(data.mag+np.log10(flux)*2.5)
                # plt.show()
                return np.stack([times, data.mag, data.err_mag], axis=-1).reshape(1, -1, 3)
            else: 
                return None
        else:
            return np.stack([times, data.mag], axis=-1).reshape(1, -1, 2)
    except:
        print('Error occurred, but continue')
        return None

def generate_random_parameter_set(u0_max=2, max_iter=100):
    ''' generate a random set of parameters. '''

    # OUTDATED: simulate -2tE to 2tE, with t0=0
    # t_0=0; t_E=50

    # new

    t_E = 10.**random.uniform(0, 1)
    t_0 = random.uniform(0 + 3 * t_E, 72 - 3 * t_E)
    f_s = 10.**random.uniform(-1, 0)

    rho = 10.**random.uniform(-4, -2) # log-flat between 1e-4 and 1e-2
    q = 10.**random.uniform(-6, 0) # including both planetary & binary events
    s = 10.**random.uniform(np.log10(0.2), np.log10(5))
    alpha = random.uniform(0, 360) # 0-360 degrees
    ## use Penny (2014) parameterization for small-q binaries ##
    if q < 1e-3:
        if q/(1+q)**2 < (1-s**4)**3/27/s**8: # close topology #
            if s < 0.1:
                uc_max = 0
            else:
                uc_max = (4+90*s**2)*np.sqrt(q/(1+s**2))/s
            xc = (s-(1-q)/s)/(1+q)
            yc = 0.
        elif s**2 > (1+q**(1/3.))**3/(1+q): # wide topology #
            uc_max = (4+min(90*s**2, 160/s**2))*np.sqrt(q)
            xc = s - 1./(1+q)/s
            yc = 0.
        else: # resonant topology
            xc, yc = 0., 0.
            uc_max = 4.5*q**0.25
        alpha_rad = alpha/180.*np.pi
        n_iter = 0
        while True:
            uc = random.uniform(0, uc_max)
            u0 = uc - xc*np.sin(alpha_rad) + yc*np.cos(alpha_rad)
            n_iter += 1
            if u0 < u0_max:
                break
            if n_iter > max_iter:
                break
    else: # for large-q binaries, use the traditional parameterization
        u0 = random.uniform(0, u0_max)
    return {
        't_0': t_0,
        't_E': t_E,
        'u_0': u0,
        'rho': rho, 
        'q': q, 
        's': s, 
        'alpha': alpha,
    }, f_s

def simulate_batch(batch_size, relative_uncertainty, time_settings_random, time_settings_even, methods, log_path, b):
    '''
    Simulate a batch of lightcurves

    Save to file:

        X_random: randomly sampled lightcurve

        X_even: evenly densely sampled lightcurve, as ground truth lightcurve

        Y: parameters

    '''
    log = open(log_path, 'a')
    time_start = time.time()
    X_random = np.empty((batch_size, time_settings_random['n_epochs'], 3))
    X_even = np.empty((batch_size, time_settings_even['n_epochs'], 2))
    Y = np.empty((batch_size, 8))
    # t_0, t_E, u_0, rho, q, s, alpha, f_s
    num_lc = 0

    print(f'Simulating batch {b}:\n' + '#'*50 + '\n')
    pbar = tqdm(total=100)
    while num_lc < batch_size:
        parameters, f_s= generate_random_parameter_set()
        Y[num_lc] = list(parameters.values()) + [f_s]

        settings_random = {
            'parameters': parameters,
            'time_settings': time_settings_random,
            'methods': methods,
        }

        settings_even = {
            'parameters': parameters,
            'time_settings': time_settings_even,
            'methods': methods,
        }

        lc_random = simulate_lc(**settings_random, 
            flux_source=1000, flux_blending=1000*(1-f_s)/f_s, relative_uncertainty=relative_uncertainty, plot=False)

        if type(lc_random) == np.ndarray:
            lc_even = simulate_lc(**settings_even, 
                flux_source=1000, flux_blending=1000*(1-f_s)/f_s, relative_uncertainty=0, plot=False)
            if type(lc_even) == np.ndarray:
                X_random[num_lc] = lc_random
                X_even[num_lc] = lc_even
                num_lc += 1

            # ABANDONED: resample
            # if num_resample > 0:
            #     for i in range(num_resample - 1):
            #         if (num_lc / batch_size * 1000) % 10 == 0:
            #             pbar.update()
            #         Y[num_lc] = list(parameters.values())
            #         lc = simulate_lc(**settings, plot=False)
            #         X[num_lc] = lc
            #         num_lc += 1
            # else:
            #     if (num_lc / batch_size * 1000) % 10 == 0:
            #         pbar.update()

            if (num_lc / batch_size * 1000) % 10 == 0:
                    pbar.update()

    pbar.close()

    with h5py.File(f'/work/hmzhao/irregular-lc/roman-{b}.h5', 'w') as opt:
        opt['X_random'] = X_random
        opt['X_even'] = X_even
        opt['Y'] = Y
    
    time_end = time.time()
    log.write(f'batch {b} stored in /work/hmzhao/irregular-lc/roman-{b}.h5, size {batch_size}, use time: {time_end - time_start}s\n')
    log.close()

        


if __name__ == '__main__':

    print('#'*50+f'\nSimulation program start at {time.time()}\n'+'#'*50)

    batch_size = int(sys.argv[1])
    num_of_batch = int(sys.argv[2])
    num_of_cpus = int(sys.argv[3])
    # num_of_resample = int(sys.argv[4])
    log_path = sys.argv[4]

    num_of_points_perlc_random = int(72. / 0.01)
    num_of_points_perlc_even = int(72. / 0.01)
    # t_0 = 0; t_E = 50; 
    t_start = 0; t_stop = 72; 
    relative_uncertainty = 0.01; 

    time_settings_random = {
            'type': 'random',
            'n_epochs': num_of_points_perlc_random,
            't_start': t_start,
            't_stop': t_stop,
        }
    time_settings_even = {
            'type': 'evenly spaced',
            'n_epochs': num_of_points_perlc_even,
            't_start': t_start,
            't_stop': t_stop,
        }
    methods = [t_start, 'VBBL', t_stop]

    log = open(log_path, 'w')
    log.write(f'Simulating {num_of_batch} batches of {batch_size} lightcurves\n'+'#'*20+'\n')
    log.close()

    pool = Pool(num_of_cpus)
    pool.map(partial(simulate_batch, batch_size, relative_uncertainty, 
                        time_settings_random, time_settings_even,
                        methods, log_path), 
                range(num_of_batch))

    print('#'*50+f'\nSimulation program end at {time.time()}\n'+'#'*50)