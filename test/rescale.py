import numpy as np
import MulensModel as mm
from tqdm import tqdm

def simulate_lc(t_0, t_E, u_0, lgrho, lgq, lgs, alpha_180, lgfs, times=None, relative_uncertainty=0, n_points=1000, orig=False):
    fs = 10**lgfs
    parameters = {
            't_0': t_0,
            't_E': t_E,
            'u_0': u_0,
            'rho': 10**lgrho, 
            'q': 10**lgq, 
            's': 10**lgs, 
            'alpha': alpha_180*180,
        }
    modelmm = mm.Model(parameters, coords=None)
    if type(times)==type(None):
        times = modelmm.set_times(t_start=parameters['t_0']-2*parameters['t_E'], t_stop=parameters['t_0']+2*parameters['t_E'], n_epochs=n_points)
    modelmm.set_magnification_methods([parameters['t_0']-2*parameters['t_E'], 'VBBL', parameters['t_0']+2*parameters['t_E']])
    magnification = modelmm.get_magnification(times)
    flux = fs * magnification + (1 - fs)
    flux *= 1 + relative_uncertainty * np.random.randn(len(flux))
    if orig:
        mag = -2.5 * np.log10(flux)
    else:
        mag = -2.5 * np.log10(flux) / 0.2
    lc = np.stack([times, mag], axis=-1)
    return lc

size = 4096 * 4

Y = np.array(np.load('./rescale_y.npy', allow_pickle=True))
pred = np.load('./rescale_pred.npy', allow_pickle=True)
for i in range(len(pred)):
    pred[i] = np.array(pred[i])

relative_uncertainty = 0.03
dchi2s = np.zeros((size,))
dparams = np.zeros((size, 6))
dparams = Y[:size] - pred[2][:size]

for i in tqdm(range(size)):
    lc_true = simulate_lc(0, 1, *Y[i].tolist(), orig=True)
    param = pred[2][i].tolist()
    lc_pred = simulate_lc(0, 1, *param, orig=True)
    dchi2 = np.sum(((lc_true[:, 1] - lc_pred[:, 1]) / (2.5 * np.log(10) * relative_uncertainty))**2)
    dchi2s[i] = dchi2

np.save('./dchi2.npy', dchi2s)
np.save('./dparams.npy', dparams)
