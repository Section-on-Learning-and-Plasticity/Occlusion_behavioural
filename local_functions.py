import numpy as np
import pandas as pd 
import math,mne
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import seaborn as sns
import rpy2.situation
for row in rpy2.situation.iter_info():
    print(row)
from rpy2.robjects import r
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

def pix_to_deg(full_size_pix,screenwith_pix=1920,screenwidth_cm=53.5,screendistance_cm=57):
    pix_per_cm = screenwith_pix/screenwidth_cm
    size_cm = full_size_pix/pix_per_cm
    dva = math.atan(size_cm/2/screendistance_cm)*2
    return np.rad2deg(dva)

def read_data(bids_dir,subjids,task = 'exp1a'):
    all_epochs = []
    for subjid in subjids:
        epochs = mne.read_epochs(f'{bids_dir}/{task}/derivatives/preprocessed/sub-S{str(subjid).zfill(2)}_{task}_eyetracking-preproc-epo.fif',preload=True)
        all_epochs.append(epochs)
        
    return all_epochs
    


def calculate_bf(res):
    from rpy2.robjects import conversion, numpy2ri

    bf_package = importr('BayesFactor')
    r = ro.r 

    # Build a combined converter: base rpy2 + numpy support
    numpy_converter = conversion.get_conversion() + numpy2ri.converter

    bf_out = []
    for i in range(res.shape[1]):
        with conversion.localconverter(numpy_converter):
            r_object = conversion.py2rpy(res[:, i])
            bf_out_r = bf_package.ttestBF(x=r_object, mu=0, rscale='medium')
            bf_out.append(np.asarray(r['as.vector'](bf_out_r))[0])

    return bf_out

    

def plot_bfs(fig, ax, x, y, n_cols=500, bf_cmap="coolwarm", exponential_minmax=10):
    bf_cols = sns.color_palette(bf_cmap, n_cols)
    val_col_map = np.logspace(-exponential_minmax, exponential_minmax, num=n_cols)
    cols_idx = [np.argmin(np.abs(val_col_map - i)) for i in y]

    # stemplot
    markerline, _, baseline = ax.stem(x, y, bottom=1, linefmt='k', markerfmt=None, basefmt=None)
    markerline.set_markerfacecolor('w')
    markerline.set_markeredgecolor('w')
    markerline.set_markersize(1)
    baseline.set_color('k')

    # markers in bf-colours
    [ax.plot(x[i], y[i], color=bf_cols[cols_idx[i]], marker='.', markersize=8, lw=0,
             markeredgecolor=bf_cols[cols_idx[i]]) for i in range(len(cols_idx))]

    # style plot
    ax.set_yscale('log')
    ax.set_ylim([10**-exponential_minmax, 10**exponential_minmax])
    ax.set_yticks([1.e-10, 1, 1.e+10])
    ax.get_xaxis().get_major_formatter().labelOnlyBase = True
    # ax.set_ylabel('BF')
    ax.set_xticks(x)

    # colorbar as inset of ax itself — coords are relative to ax, no figure transforms needed
    import matplotlib as mpl
    cax = ax.inset_axes([-0.03, 0.0, 0.03, 1.0])  # [x0, y0, width, height] in axes fractions
    norm = mpl.colors.LogNorm(vmin=10**-exponential_minmax, vmax=10**exponential_minmax)
    bf_cmap_obj = sns.color_palette(bf_cmap, as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=bf_cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([])



def shrink_axes(ax,percent_X=0.9,percent_Y=0.9):
    box = ax.get_position()  # Get the current position of the subplot
    ax.set_position([box.x0+(box.width * (1-percent_X))/2, box.y0+(box.height * (1-percent_Y))/2, box.width * percent_X, box.height * percent_Y]) 
    return ax

def make_subplots(ax,bottom=0,height="30%"):
    ax_in = inset_axes(ax,
        width="100%",  
        height=height,  
        bbox_to_anchor=(0.05, bottom, 0.9, 1.0),  # 
        bbox_transform=ax.transAxes, 
        loc='lower center') 
    return ax_in

