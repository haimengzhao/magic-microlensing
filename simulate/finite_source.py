import numpy as np
import MulensModel as mm
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

from multiprocessing import Pool
from functools import partial
import sys, time

def load_data(filename, n_chunks=25):
    with h5py.File(filename, 'r') as f:
        X = f['X'][...]
        Y = f['Y'][...]
    X = X[:, :, :2]
    X_gap = add_gap(X, n_chunks=n_chunks)
    return X_gap, Y

def save_data(filename, X, Y_rho, dchi2):
    with h5py.File(filename, 'w') as f:
        f['X'] = X
        f['Y'] = Y_rho
        f['dchi2'] = dchi2

def add_gap(X, n_chunks=25):
    gap_len = X.shape[1] // n_chunks
    gap_left = np.random.randint(0, X.shape[1]-gap_len, (len(X),))
    X_gap = np.zeros((X.shape[0], X.shape[1]-gap_len, X.shape[2]))
    for i in range(len(X)):
        left, gap, right = np.split(X[i], [gap_left[i], gap_left[i] + gap_len], axis=0)
        lc = np.vstack([left, right])
        X_gap[i] = lc
    return X_gap

def simulate_lc(t_0, t_E, u_0, rho, q, s, alpha, fs, relative_uncertainty=0, n_points=500, times=None,
                point_source=False, return_times=False):
    time_settings = {
            'type': 'random',
            'n_epochs': n_points,
            't_start': t_0-2*t_E,
            't_stop': t_0+2*t_E,
        }
    if times is None:
        raw = np.random.rand(time_settings['n_epochs'])
        dt = time_settings['t_stop'] - time_settings['t_start']
        times = time_settings['t_start'] + np.sort(raw) * dt
    if point_source:
        parameters = {
            't_0': t_0,
            't_E': t_E,
            'u_0': u_0,
            'q': q, 
            's': s, 
            'alpha': alpha,
        }
    else:
        parameters = {
            't_0': t_0,
            't_E': t_E,
            'u_0': u_0,
            'rho': rho, 
            'q': q, 
            's': s, 
            'alpha': alpha,
        }
    modelmm = mm.Model(parameters, coords=None)
    # times = modelmm.set_times(t_start=parameters['t_0']-2*parameters['t_E'], t_stop=parameters['t_0']+2*parameters['t_E'], n_epochs=n_points)
    if point_source:
        modelmm.set_magnification_methods([parameters['t_0']-2*parameters['t_E'], 'point_source', parameters['t_0']+2*parameters['t_E']])
    else:
        modelmm.set_magnification_methods([parameters['t_0']-2*parameters['t_E'], 'VBBL', parameters['t_0']+2*parameters['t_E']])
    magnification = modelmm.get_magnification(times)
    flux = 1000 * (magnification + (1-fs)/fs)
    flux *= 1 + relative_uncertainty * np.random.randn(len(flux))
    mag = (22 - 2.5 * np.log10(flux))
    lc = np.stack([times, mag], axis=-1)
    if return_times:
        return lc, times
    return lc

def cal_dchi2(X, Y, relative_uncertainty = 0.03):
    dchi2 = np.empty((len(X),))
    X = np.empty_like(X)
    Y_rho = np.empty_like(Y)
    pbar = tqdm(total=100)
    for i in range(len(X)):
        params = Y[i].copy()
        params[3] = 10.**np.random.uniform(-4, -2)
        times = X[i, :, 0]
        data = simulate_lc(*params, relative_uncertainty=relative_uncertainty, times=times)
        lc = simulate_lc(*params, relative_uncertainty=0, times=times)
        lc_ps = simulate_lc(*params, relative_uncertainty=0, point_source=True, times=times)
        chi2 = np.sum(((data[:, 1] - lc[:, 1]) / (2.5 * np.log(10) * relative_uncertainty))**2)
        chi2_ps = np.sum(((data[:, 1] - lc_ps[:, 1]) / (2.5 * np.log(10) * relative_uncertainty))**2)
        dchi2[i] = chi2_ps - chi2
        Y_rho[i] = params
        X[i] = data
        if (i / len(X) * 1000) % 10 == 0:
            pbar.update()
    pbar.close()
    return X, Y_rho, dchi2

def process_data(i):
    X, Y = load_data(f'/work/hmzhao/irregular-lc/KMT-{i}-fixrho-mp.h5')
    X, Y_rho, dchi2 = cal_dchi2(X, Y)
    save_data(f'/work/hmzhao/irregular-lc/KMT-{i}-rhodchi2-mp.h5', X, Y_rho, dchi2)
    print(f'/work/hmzhao/irregular-lc/KMT-{i}-rhodchi2-mp.h5 saved')

if __name__ == '__main__':
    num_of_batch = int(sys.argv[1])
    num_of_cpus = int(sys.argv[2])

    print('#'*50+f'\nSimulation program start at {time.time()}\n'+'#'*50)

    pool = Pool(num_of_cpus)
    pool.map(process_data, list(range(num_of_batch)))

    print('#'*50+f'\nSimulation program end at {time.time()}\n'+'#'*50)



