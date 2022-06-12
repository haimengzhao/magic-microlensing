import numpy as np
import matplotlib.pyplot as plt
import pandas
import VBBinaryLensing
from scipy.optimize import minimize, fmin

def input_data(lc_file, mdn_file):
    lc_data = np.loadtxt(lc_file, delimiter=',')
    time = lc_data[:, 0]
    mag = lc_data[:, 1] + 18
    merr = np.ones_like(mag) * 0.033
    flux = 10**(0.4*(18-mag))
    ferr = merr*flux*np.log(10)/2.5
    lc = pandas.DataFrame({'time':time, 'flux':flux, 'ferr':ferr, 'mag':mag, 'merr':merr})

    ## import mdn data ##
    mdn_data = np.load(mdn_file)
    weight = mdn_data['pi']
    descending_order = np.argsort(weight)[::-1]
    weight = weight[descending_order]
    mean = mdn_data['loc'][descending_order]
    dispersion = mdn_data['scale'][descending_order]
    mdn = pandas.DataFrame(np.vstack((weight, mean.T)).T, columns=('weight', 'u0', 'lgq', 'lgs', 'a/180', 'lgfs'))
#    print(weight)
#    print(mean)
#    print(mdn)
    return lc, mdn

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


def compute_model_lc(time_array, fitting_parameters, VBBL):
    u0, lgq, lgs, ad180 = fitting_parameters
    q, s = 10**lgq, 10**lgs
    alpha = ad180 * np.pi # convert to radian
    t0, te, rho = 0, 1, 1e-3
    tau = (time_array-t0)/te
    xs = tau*np.cos(alpha) - u0*np.sin(alpha)
    ys = tau*np.sin(alpha) + u0*np.cos(alpha)
    magnifications = np.array([VBBL.BinaryMag2(s, q, xs[i], ys[i], rho) for i in range(len(xs))])
    return magnifications

def perform_optimization(time, flux, ferr, para_initial):
    VBBL = VBBinaryLensing.VBBinaryLensing()
    
    def compute_chisq(fitting_parameters, time, flux, ferr, VBBL, return_model=False):
        magnifications = compute_model_lc(time, fitting_parameters, VBBL)
        chi2, fs, fb, fserr, fberr = get_fsfb(magnifications, flux, ferr)
        if return_model:
            return chi2, fs, fb
        return chi2
    
#    result = minimize(compute_chisq, para_initial, args=(time, flux, ferr, VBBL))
#    para_best = result.x
#    chi2_min = result.fun
#    print(result.success, para_best, chi2_min)

    para_best, chi2_min, iter, funcalls, warnflag, allevcs = fmin(compute_chisq, para_initial, args=(time, flux, ferr, VBBL), full_output=True, retall=True, maxiter=1000, maxfun=5000)

#    chi2_min, fs, fb = compute_chisq(para_initial, time, flux, ferr, VBBL, return_model=True)
    print('initial chisq: ', chi2_min)
    chi2_min, fs, fb = compute_chisq(para_best, time, flux, ferr, VBBL, return_model=True)
    print('best chisq & (fs, fb): ', chi2_min, fs, fb)
    time_model = np.arange(-2, 2, 0.001)
#    magnifications = compute_model_lc(time_model, para_initial, VBBL)
    magnifications = compute_model_lc(time_model, para_best, VBBL)
    mag_model = 18 - 2.5*np.log10(magnifications*fs + fb)
    model = np.vstack((time_model, mag_model))
    return para_best, chi2_min, model, warnflag

def main():
    lc, mdn = input_data('./lc_icml.csv', 'gm_30.npz')
    best_parameters = []
#    mdn['chi2_min'] = None
#    mdn['fmin_flag'] = None
#    mdn['u0_best'] = None
#    mdn['lgq_best'] = None
#    mdn['lgs_best'] = None
#    mdn['alpha/180_best'] = None
    for index in mdn.index:
        para_initial = mdn.loc[index, ['u0', 'lgq', 'lgs', 'a/180']].values
        print(para_initial)
        para_best, chi2_min, model, warnflag = perform_optimization(lc['time'].values, lc['flux'].values, lc['ferr'].values, para_initial)
        best_parameters.append(np.hstack((chi2_min, warnflag, para_best)))

        plt.plot(model[0], model[1], label=r'$\pi$=%.2f, $\chi^2$=%.1f'%(mdn.at[index, 'weight'], chi2_min))
        if index >=4:
            break

    best_parameters = np.array(best_parameters)
    mdn_add = pandas.DataFrame(best_parameters, columns=('chi2_min', 'fmin_flag', 'u0_best', 'lgq_best', 'lgs_best', 'alpha/180_best'))
    mdn = mdn.join(mdn_add, how='outer')
    print(mdn)
    plt.errorbar(lc['time'], lc['mag'], yerr=lc['merr'], ls='none', marker='o', mfc='none')
    plt.gca().invert_yaxis()
    plt.legend(loc=0)
    plt.xlim(-1, 1)
    plt.savefig('lc_icml.png')
    # plt.show()
    return

main()