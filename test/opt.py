import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas
import VBBinaryLensing
from scipy.optimize import minimize, fmin

from multiprocessing import Pool
import timeout_decorator

from tqdm import tqdm
import time

class MinimizeStopper(object):
    def __init__(self, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()
    def __call__(self, xk=None):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            raise TimeoutError()


def get_fsfb(amp, flux, ferr):
    sig2 = ferr**2
    wght = flux/sig2
    d = np.ones(2)
    d[0] = np.sum(wght*amp)
    d[1] = np.sum(wght)
    b = np.zeros((2,2))
    b[0,0] = np.sum(amp**2/sig2)
    b[0,1] = np.sum(amp/sig2)
    b[1,0] = b[0,1]
    b[1,1] = np.sum(1./sig2)
    c = np.linalg.inv(b)
    fs = np.sum(c[0]*d)
    fb = np.sum(c[1]*d)
    fserr = np.sqrt(c[0,0])
    fberr = np.sqrt(c[1,1])
    fmod = fs*amp+fb
    chi2 = np.sum((flux-fmod)**2/sig2)
    return chi2,fs,fb,fserr,fberr

def binary_magnification(s, q, xs, ys, rho, VBBL):
    if s < 20 and q > 1e-8 and s > 0.01 and q < 1e3:
        return VBBL.BinaryMag2(s, q, xs, ys, rho)
    else:
        u2 = xs**2 + ys**2
        return (u2+2)/np.sqrt(u2*(u2+4))

def compute_model_lc(time_array, fitting_parameters, VBBL):
    if len(fitting_parameters) == 6:
        u0, lgq, lgs, ad180, t0, te = fitting_parameters
        rho = 1e-3
    else:
        u0, lgq, lgs, ad180 = fitting_parameters
        t0, te, rho = 0, 1, 1e-3
    q, s = 10**lgq, 10**lgs
    alpha = ad180 * np.pi # convert to radian
    tau = (time_array-t0)/te
    xs = tau*np.cos(alpha) - u0*np.sin(alpha)
    ys = tau*np.sin(alpha) + u0*np.cos(alpha)
    magnifications = np.array([binary_magnification(s, q, xs[i], ys[i], rho, VBBL) for i in range(len(xs))])
    return magnifications

# @timeout_decorator.timeout(60, use_signals=True)
def perform_optimization(time, flux, ferr, para_initial, verbose=True):
    VBBL = VBBinaryLensing.VBBinaryLensing()
    
    def compute_chisq(fitting_parameters, time, flux, ferr, VBBL, return_model=False):
        magnifications = compute_model_lc(time, fitting_parameters, VBBL)
        chi2, fs, fb, fserr, fberr = get_fsfb(magnifications, flux, ferr)
        if return_model:
            return chi2, fs, fb
        return chi2

    para_best, chi2_min, iter, funcalls, warnflag, allvecs = fmin(compute_chisq, para_initial, args=(time, flux, ferr, VBBL), full_output=True, retall=True, maxiter=1000, maxfun=1000, disp=verbose, callback=MinimizeStopper(max_sec=60))

    chi2_min, fs, fb = compute_chisq(para_initial, time, flux, ferr, VBBL, return_model=True)
    if verbose:
        print('initial chisq: ', chi2_min)
    chi2_min, fs, fb = compute_chisq(para_best, time, flux, ferr, VBBL, return_model=True)
    if verbose:
        print('best chisq & (fs, fb): ', chi2_min, fs, fb)
    time_model = np.arange(-2, 2, 0.001)
#    magnifications = compute_model_lc(time_model, para_initial, VBBL)
    magnifications = compute_model_lc(time_model, para_best, VBBL)
    mag_model = 18 - 2.5*np.log10(magnifications*fs + fb)
    model = np.vstack((time_model, mag_model))
    return para_best, chi2_min, model, warnflag

def get_best_params(lc, locs, n_gau, verbose=False, message=None, opt_t=False):
    size = len(lc)
    if opt_t:
        best_parameters = np.zeros((size, n_gau, 8))
    else:
        best_parameters = np.zeros((size, n_gau, 6))
    for i in tqdm(range(size)):
        lc_i = lc[i]
        ind_unique = np.unique(lc_i[:, 0], return_index=True)[1] 
        lc_i = lc_i[ind_unique]
        for index in range(n_gau):
            para_initial = locs[i, index, :-1] # (u0, lgq, lgs, ad180)
            if verbose:
                print(para_initial)
            try:
                # if not opt_t:
                para_best, chi2_min, model, warnflag = perform_optimization(lc_i[:, 0], lc_i[:, 2], lc_i[:, 3], para_initial, verbose=verbose)
                if opt_t:
                    para_best = np.concatenate((para_best, [0, 1])) # (u0, lgq, lgs, ad180, t0, te)
                    para_best, chi2_min, model, warnflag = perform_optimization(lc_i[:, 0], lc_i[:, 2], lc_i[:, 3], para_best, verbose=verbose)
            except:
                print('timeout')
                para_best = np.ones_like(para_initial) * np.nan
                if opt_t:
                    para_best = np.concatenate((para_best, [np.nan, np.nan]))
                chi2_min = np.inf
                warnflag = -1
            best_parameters[i, index] = np.hstack((chi2_min, warnflag, para_best))
    print(f'# {message} done!')
    return best_parameters


if __name__ == '__main__':

    opt_t = False

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    input_file = np.load(input_filename)
    lc = input_file['lc']
    locs = input_file['locs']

    verbose = False

    n_gau = 12
    size = 4096 * 4
    if opt_t:
        best_parameters = np.zeros((size, n_gau, 8))
    else:
        best_parameters = np.zeros((size, n_gau, 6))
    

    n_processes = 64
    step = size // n_processes
    pool = Pool(processes=n_processes)
    results = []

    for i in tqdm(range(n_processes)):
        results.append(pool.apply_async(get_best_params, args=(lc[i*step:(i+1)*step], locs[i*step:(i+1)*step], n_gau, verbose, f'{i*step}:{(i+1)*step}', opt_t)))

    pool.close()
    pool.join()

    for i, result in enumerate(results):
        best_parameters[i*step:(i+1)*step] = result.get()
    
    np.save(output_filename, best_parameters)