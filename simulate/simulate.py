"""
from https://github.com/rpoleski/MulensModel/blob/master/examples/example_18_simulate.py
and https://github.com/LRayleighJ/MDN_lc_iden/blob/main/simudata/gen_simu.py

BinaryJax from https://github.com/CoastEgo/BinaryJax

Script for simulating microlensing lightcurves.

NB: output mag = 22 - 2.5lg(flux)
"""
import os
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import sys

import MulensModel as mm
from BinaryJax import model
os.environ["CUDA_VISIBLE_DEVICES"]="" 

import random
import h5py
from tqdm import tqdm
import time 
from multiprocessing import Pool
from functools import partial

from scipy.stats import truncnorm


def simulate_lc(
        parameters, n_points, t_start, t_stop,
        relative_uncertainty=0.01
        ):
    """
    Simulate and save light curve.

    Parameters :
        parameters: *dict*
            Parameters of the model - keys are in MulensModel format, 
            t_0, t_E, u_0, rho, q, s, alpha, f_s.
            
        n_points: *int*
            Number of points to simulate.
        
        t_start: *float*
            Start time of the observation.
        
        t_stop: *float*
            End time of the observation.

        relative_uncertainty: *float*
            Relative uncertainty of the simulated data (this is close to
            sigma in magnitudes).

    """
    # try:
    raw = np.random.rand(n_points)
    dt = t_stop - t_start
    times = t_start + np.sort(raw) * dt

        # magnification = model.get_magnification(times)

        # flux = flux_source * magnification + flux_blending
        # flux_err = relative_uncertainty * flux

        # flux *= 1 + relative_uncertainty * np.random.randn(len(flux))

        # data = mm.MulensData([times, flux, flux_err], phot_fmt='flux')
        
    def microlensing_model(t_0, t_E, u_0, rho, q, s, alpha, f_s, times):
        dic = {'t_0': t_0, 'u_0': u_0, 't_E': t_E, 'rho': rho, 'q': q, 's': s, 'alpha_deg': alpha, 'times': times, 'retol': 1e-2}
        magnification = model(**dic)
        flux = 1000 * magnification + 1000 * (1-f_s)/f_s
        magnitudes = -2.5 * jnp.log10(flux)
        return magnitudes

    mag_model = microlensing_model(**parameters, times=times)
    mag_errors = jnp.ones_like(mag_model) * relative_uncertainty
    mag_data = np.random.normal(mag_model, mag_errors)
    data = mm.MulensData([times, mag_data, mag_errors], phot_fmt='mag')

    ## compute the gradients ##
    microlensing_model_grad = jax.jacfwd(microlensing_model, argnums=(0, 1, 2, 3, 4, 5, 6, 7))
    mag_grad = jnp.array(microlensing_model_grad(parameters['t_0'], parameters['t_E'], parameters['u_0'], parameters['rho'], parameters['q'], parameters['s'], parameters['alpha'], parameters['f_s'], times))

    ## compute the Fisher matrix ##
    # ndim = mag_grad.shape[0]
    # Fmat = jnp.zeros((ndim, ndim))
    # for i in range(ndim):
    #     for j in range(ndim):
    #         Fmat[i, j] = jnp.sum(mag_grad[i] * mag_grad[j]/mag_errors**2)
    # einsum
    Fmat = jnp.einsum('in,jn->ij', mag_grad, mag_grad/mag_errors**2)
    
    single = mm.Model({'t_0': parameters['t_0'], 'u_0': parameters['u_0'], 't_E': parameters['t_E']})
    event_single = mm.Event([data], single)
    chi2 = event_single.get_chi2()
    # print("chi^2 single: {:.2f}".format(chi2))

    if chi2 > 1000:
        # plt.plot(data.mag+np.log10(flux)*2.5)
        # plt.show()
        return np.stack([times, data.mag, data.err_mag], axis=-1).reshape(1, -1, 3), Fmat
    else: 
        return None, None
    # except:
    #     print('Error occurred, but continue')
    #     return None, None

def generate_random_parameter_set(u0_max=1, max_iter=100):
    ''' generate a random set of parameters. '''

    # fix t_0, t_E
    t_0 = 0.; t_E = 1.

    # random t_0, t_E
    # t_E = 10**truncnorm.rvs((np.log10(5)-1.15)/0.45, (np.log10(100)-1.15)/0.45, loc=1.15, scale=0.45)
    # t_0 = random.uniform(-t_E, t_E)
    
    # f_s
    f_s = 10.**random.uniform(-1, 0)

    # rho = 10.**random.uniform(-4, -2) # log-flat between 1e-4 and 1e-2
    rho = 10.**(-3)
    q = 10.**random.uniform(-3, 0) # including both planetary & binary events
    s = 10.**random.uniform(np.log10(0.3), np.log10(3))
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
        'f_s': f_s,
    }

def simulate_batch(batch_size, relative_uncertainty, n_points, t_start, t_stop, log_path, save_path, b):
    '''
    Simulate a batch of lightcurves

    Save to file:

        X: lightcurve

        Y: parameters
        
        F: Fisher matrices

    '''
    log = open(log_path, 'a')
    time_start = time.time()
    X = np.empty((batch_size, n_points, 3))
    Y = np.empty((batch_size, 8))
    F = np.empty((batch_size, 8, 8))
    # t_0, t_E, u_0, rho, q, s, alpha, f_s
    num_lc = 0

    print(f'Simulating batch {b}:\n' + '#'*50 + '\n')
    log.write(f'Simulating batch {b}:\n' + '#'*50 + '\n')
    # log save_path
    print(f'Saving to {save_path}\n')
    log.write(f'Saving to {save_path}\n')
    
    pbar = tqdm(total=100)
    while num_lc < batch_size:
        # print(num_lc)
        parameters= generate_random_parameter_set()
        Y[num_lc] = list(parameters.values())

        lc, Fmat = simulate_lc(parameters, n_points, t_start, t_stop, relative_uncertainty)

        if type(lc) == np.ndarray:
            X[num_lc] = lc
            F[num_lc] = Fmat
            num_lc += 1

        if (num_lc / batch_size * 1000) % 10 == 0:
            pbar.update()

    pbar.close()

    with h5py.File(save_path + f'lc-{b}.h5', 'w') as opt:
        opt['X'] = X
        opt['Y'] = Y
        opt['F'] = F
    
    time_end = time.time()
    log.write(f'batch {b} stored, size {batch_size}, use time: {time_end - time_start}s\n')
    log.close()

        


if __name__ == '__main__':

    print('#'*50+f'\nSimulation program start at {time.time()}\n'+'#'*50)

    batch_size = int(sys.argv[1])
    num_of_batch = int(sys.argv[2])
    num_of_cpus = int(sys.argv[3])
    log_path = './log.log'
    save_path = '/work/hmzhao/data/'
    os.makedirs(save_path, exist_ok=True)

    n_points = int(500)
    t_start = -2; t_stop = 2; 
    relative_uncertainty = 0.03; 
    
    log = open(log_path, 'w')
    log.write(f'Simulating {num_of_batch} batches of {batch_size} lightcurves\n'+'#'*20+'\n')
    log.close()

    # pool = Pool(num_of_cpus)
    # pool.map(partial(simulate_batch, batch_size, relative_uncertainty, 
    #             n_points, t_start, t_stop, log_path, save_path),
    #          range(num_of_batch))
    simulate_batch(batch_size, relative_uncertainty, 
                n_points, t_start, t_stop, log_path, save_path, 0)

    print('#'*50+f'\nSimulation program end at {time.time()}\n'+'#'*50)