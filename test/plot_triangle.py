import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import matplotlib.cm as cm

def hist2d(x, y, *args, **kwargs):
    """
    Plot a 2-D histogram of samples.

    """
    ax = kwargs.pop("ax", plt.gca())

    extent = kwargs.pop("extent", [[x.min(), x.max()], [y.min(), y.max()]])
    bins = kwargs.pop("bins", 30)
    color = kwargs.pop("color", "b")
    linewidths = kwargs.pop("linewidths", None)
    plot_datapoints = kwargs.get("plot_datapoints", True)
    plot_contours = kwargs.get("plot_contours", True)

    cmap = cm.get_cmap("gray")
    cmap._init()
    cmap._lut[:-3, :-1] = 0.
    cmap._lut[:-3, -1] = np.linspace(1, 0, cmap.N)

    X = np.linspace(extent[0][0], extent[0][1], bins + 1)
    Y = np.linspace(extent[1][0], extent[1][1], bins + 1)
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=(X, Y),
                                 weights=kwargs.get('weights', None))
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "`extent` argument.")

    V = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2)
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]

    for i, v0 in enumerate(V):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]

    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    X, Y = X[:-1], Y[:-1]

    if plot_datapoints:
        ax.plot(x, y, "o", color=color, ms=1.5, zorder=-1, alpha=0.1,
                rasterized=True)
        if plot_contours:
            ax.contourf(X1, Y1, H.T, [V[-1], H.max()],
                        cmap=LinearSegmentedColormap.from_list("cmap",
                                                               ([1] * 3,
                                                                [1] * 3),
                        N=2), antialiased=False)

    if plot_contours:
#        ax.pcolor(X, Y, H.max() - H.T, cmap=cmap)
        V = [V[-1],V[-2],V[-3]]
        ax.contour(X1, Y1, H.T, V, colors=color, linewidths=linewidths)
#        ax.contourf(X1, Y1, H.T, [V[-1], H.max()], cmap=LinearSegmentedColormap.from_list("cmap",([1] * 3,[1] * 3),N=2), antialiased=False)

    data = np.vstack([x, y])
    mu = np.mean(data, axis=1)
    cov = np.cov(data)
    if kwargs.pop("plot_ellipse", False):
        Ellipse(mu, cov, ax=ax, edgecolor="r", ls="dashed")

#    ax.set_xlim(extent[0])
#    ax.set_ylim(extent[1])
    return


def plot_covariance(params,labels,cov_mat,extents=None,chain=None,fig=None,axes=None,weight=1.0,ground_truth=None,ground_truth_color='r', ground_truth_ls='dashed'):
    ''' plot covariance matrix: both theoretical & mcmc chain. '''
    K = len(params)
    if axes is None:
        ## set up axes ##
        factor = 2.0           # size of one side of one panel
        lbdim = 1.2 * factor   # size of left/bottom margin
        trdim = 0.15 * factor  # size of top/right margin
        whspace = 0.1         # w/hspace size
        plotdim = factor * K + factor * (K - 1.) * whspace
        dim = lbdim + plotdim + trdim
        fig,axes = plt.subplots(K,K,figsize=(8.5,8.5))
        lb = lbdim / dim
        tr = (lbdim + plotdim) / dim
        fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace)
    ## set up axex extent ##
    if extents is None:
        extents = [[params[i]-3*np.sqrt(cov_mat[i, i]), params[i]+3*np.sqrt(cov_mat[i, i])] for i in range(len(params))]
#    else:
#        extents = [[x.min(), x.max()] for x in chain.T]
    ##
    for i in range(K):
        ax = axes[i,i]
        mu_x,sigma_x = params[i],np.sqrt(cov_mat[i,i])
        x = np.linspace(extents[i][0],extents[i][1],100)
        p = 1/np.sqrt(2*np.pi)/sigma_x * np.exp(-(x-mu_x)**2/2./sigma_x**2)
        p*= weight
        ax.plot(x,p,color='%f'%(1-weight))
        if not(chain is None):
            ax.hist(chain[:,i],histtype='step',density=1)
        ax.set_xlim(extents[i])
        if not(ground_truth is None):
            ax.axvline(ground_truth[i], color=ground_truth_color, linestyle=ground_truth_ls)
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if i < K-1:
            ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                ax.set_xlabel(labels[i])
                ax.xaxis.set_label_coords(0.5, -0.7)
        for j in range(K):
            ax = axes[i,j]
            if j > i:
                ax.set_visible(False)
                ax.set_frame_on(False)
                continue
            elif j == i:
                continue
            ax.set_xlim(extents[j])
            ax.set_ylim(extents[i])
            ## plot error ellipse from given covariance matrix ##
            mu_y,sigma_y = params[j],np.sqrt(cov_mat[j,j])
            sigx2,sigy2,sigxy = cov_mat[i,i],cov_mat[j,j],cov_mat[i,j]
            ## find principle axes ##
            sig12 = 0.5*(sigx2+sigy2) + np.sqrt((sigx2-sigy2)**2*0.25+sigxy**2)
            sig22 = 0.5*(sigx2+sigy2) - np.sqrt((sigx2-sigy2)**2*0.25+sigxy**2)
            sig1 = np.sqrt(sig12)
            sig2 = np.sqrt(sig22)
            alpha = 0.5*np.arctan(2*sigxy/(sigx2-sigy2))
            if sigy2 > sigx2:
                alpha += np.pi/2.
            ## plot ellipse ##
            t = np.linspace(0,2*np.pi,300)
            x = mu_x + sig1*np.cos(t)*np.cos(alpha) - sig2*np.sin(t)*np.sin(alpha)
            y = mu_y + sig1*np.cos(t)*np.sin(alpha) + sig2*np.sin(t)*np.cos(alpha)
#            ax.plot(y,x,'k')
            ax.plot(y,x,color='%f'%(1-weight))
            ax.plot(mu_y,mu_x,'+',color='%f'%(1-weight))
            if not(ground_truth is None):
                ax.plot(ground_truth[j], ground_truth[i], marker='+', color=ground_truth_color)
            ## plot error ellipse from mcmc chain ##
            if not(chain is None):
                hist2d(chain[:,j],chain[:,i],ax=ax,plot_contours=True,plot_datapoints=False)
#                hist2d(chain[:,j],chain[:,i],ax=ax,extent=[extents[j],extents[i]],plot_contours=True,plot_datapoints=False)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            if i < K-1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j])
                    ax.xaxis.set_label_coords(0.5,-0.7)
            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.6,0.5)
#    plt.show()
    return fig,axes

