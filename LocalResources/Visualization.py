"""
A file containing case visualtization functions. We have stripped the drug plots out due to limitations on which data are shared--only those essential for classification.

jha
"""

#import modules
import os
import pickle
from time import time
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter
import seaborn as sns
#
from . import NSRL_resources as nsrl
from . import turbo_colormap_mpl
from . import ConsciousnessClassifier as cc
from . import PlotOptions as plo
from . import tol_colors

##############
### GLOBAL ###
##############

# Params
#DIR_RAW_HV = #
DIR_DATA_HV = 'Data/Volunteer/'
DIR_DATA_OR = 'Data/OR/'
DIR_INFTBL_HV = 'Results/Volunteer Inference Tables/'
DIR_INFTBL_OR = 'Results/OR Inference Tables/'


CLASSIFIER_NAMES = ['Sdb','PCA', 'LDA+HMM2']

# figure geometry specifications

FIG_SIZE=[3.5,3.5]

GRID_N_ROWS = 4
GRID_N_COL = 2
GRID_WIDTHS = [15,1]
GRID_HEIGHTS = [15,15,12,1]

AXES_TO_ALIGN=[0,2,3,4] # list of axes to be aligned (excludes colorbar and legends)

# Preconditions
assert os.path.exists(DIR_DATA_HV)
assert os.path.exists(DIR_DATA_OR)
assert os.path.exists(DIR_INFTBL_HV)
assert os.path.exists(DIR_INFTBL_OR)

def timeseries_hv(case_id):
    """plot single case timeseries and model performance for healthy volunteer data
    models to be included in performance are a global variable
    layout of plot is also a global variable
    
    Parameters
    ----------
    case_id : str
        2 char string of unique case id
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        figure with spgm, efefctsite concentration, model performance, and labels
    alp : float
        alpha value for performance
    """
    # Load  data
    data_dict = cc.load_volunteer_todict(DIR_DATA_HV, case_id)
    #drug_times, drug_conc = _load_volunteer_drug(case_id, DIR_RAW_HV)
    inference_table  = _filter_inftbl(case_id,CLASSIFIER_NAMES,DIR_INFTBL_HV)


    #organize plot
    fig = plt.figure(figsize=FIG_SIZE)
    plt.title(f'Internal Validation: Case {case_id}')
    gs = GridSpec(GRID_N_ROWS,GRID_N_COL, height_ratios=GRID_HEIGHTS ,width_ratios=GRID_WIDTHS)
    gs = plt_spgm(data_dict,gs)
    #gs = plt_drugs_vol(drug_times,drug_conc,gs)
    gs = plt_performance(inference_table,CLASSIFIER_NAMES,gs,0.4)
    gs = plt_label(inference_table,gs)
    fig = _align_axes(fig)
    #plt.tight_layout()
    return fig

def timeseries_or(case_id):
    """plot single case timeseries and model performance for OR data
    models to be included in performance are a global variable
    layout of plot is also a global variable
    
    Parameters
    ----------
    case_id : str
        string of unique case id
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        figure with spgm, relevant bolus & infusion, model performance, and labels
    """

    # Load  data
    data_dict = cc.load_OR_todict(DIR_DATA_OR, case_id)
    inference_table  = _filter_inftbl(case_id,CLASSIFIER_NAMES,DIR_INFTBL_OR)

    #organize plot
    fig = plt.figure(figsize=FIG_SIZE)
    gs = GridSpec(GRID_N_ROWS,GRID_N_COL, height_ratios=GRID_HEIGHTS ,width_ratios=GRID_WIDTHS)
    gs = plt_spgm(data_dict,gs)
    #gs = plt_drugs_or(data_dict,gs)
    gs = plt_performance(inference_table,CLASSIFIER_NAMES,gs,0.3)
    gs = plt_label(inference_table,gs)
    fig = _align_axes(fig)
    # plt.tight_layout(**plo.layout_pad)
    return fig

def plt_spgm(data_dict, gridspec):
    """plot spectogram on
    x axis is in minutes  since EEG on
    
    Parameters
    ----------ax_lgd
        grid of overall figure to plot
        spectogram plotted at [0,0] with the color bar at [0,1]

    """
    ax_spgm = plt.subplot(gridspec[0,0])
    ax_spgm_cbar = plt.subplot(gridspec[0,1])
    cbar = ax_spgm.pcolormesh(data_dict['t']/60, data_dict['f'], 
        data_dict['Sdb'], cmap='magma', vmin=-10,#percentiles[0], 
        vmax=30, zorder=-10)#percentiles[1])
    
    plt.colorbar(cbar, ax_spgm_cbar, ticks=[-10,0,10,20,30])
    ax_spgm_cbar.set_ylabel('Power (dB)')
    ax_spgm.set_ylabel('Freq (Hz)')
    ax_spgm.set_ylim([0,50])
    ax_spgm.set_yticks([0,10,20,30,40,50])
    ax_spgm.set_yticklabels([0,'','','','',50])
    #ax_spgm.set_xticklabels([])
    ax_spgm.set_rasterization_zorder(-5)

    return gridspec

def plt_drugs_vol(drug_times,drug_conc,gridspec):
    """plotting fuction for volunteer effectsite concentation
    
    Parameters
    ----------
    drug_times : array
        time ion mimn
    drug_conc : array
        effect site concentration in ug/ml
    gridspec : matplotlib.gridspec.GridSpec
        grid of overall figure to plot
        drugs plot will be at [1,0]
    
    Returns
    -------
    matplotlib.gridspec.GridSpec
        grid of overall figure to plot
        drugs plot at [1,0]
    """
    ax_drugs = plt.subplot(gridspec[1,0])
    ax_drugs.plot(np.hstack([0, drug_times/60]),
                  np.hstack([0, drug_conc]) ,color='k')
    ax_drugs.set_ylabel('Propofol Effect Site Concentration \n(mcg/ml)')
    ax_drugs.set_xticklabels([])
    ax_drugs.set_ylim([0,6.])
    plt.legend()
    return gridspec

def plt_performance(case_inference_table,classifier_names,gridspec, alp):
    """plotting function for model performance
    x axis is in minutes since EEG on
    
    Parameters
    ----------
    case_inference_table : pandas.DataFrame
        model inference for each spgm window for a single case
        expects columns of 'times', '{classifier_name}_py'
        all classifiers to be plotted must have predictions as a column called '{classifier_name}_py'
    classifier_names : list
        names of classifiers for for which models to plot  
    gridspec : matplotlib.gridspec.GridSpec
        grid of overall figure to plot
        performance plot will be at [2,0]
        
    
    Returns
    -------
    matplotlib.gridspec.GridSpec
        grid of overall figure to plot
        performance plot at [2,0]

    """
    first_ax = plt.gcf().axes[0]
    # plotting specs
    plot_markers = ['.', '.', '.']
    plot_colors = tol_colors.tol_cset('bright')[:3]
    
    ax_perf = plt.subplot(gridspec[2,0],sharex=first_ax)

    for idx, cname in enumerate(classifier_names):
        times = case_inference_table.times.values
        py = case_inference_table[cname+'_py']
        # plot classifier predictions
        ax_perf.scatter(times/60, py, marker=plot_markers[idx], c=[plot_colors[idx]], 
                label=cname, alpha=alp, zorder=-10)

    ax_perf.set_ylabel('P(y)')
    ax_perf.set_rasterization_zorder(-5)
    plt.legend()
    return gridspec

def plt_label(case_inference_table,gridspec):
    """plot 1D heatmap of state labels
    x axis is in minutes since EEG on
    
    Parameters
    ----------
    case_inference_table : pandas.DataFrame
        model prediction for each spectogram window for a single case
        expects columns with 'l' and 'times'
    
    gridspec : matplotlib.gridspec.GridSpec
        grid of overall figure to plot
        performance plot will be at [3,0]
    
    Returns
    -------
    matplotlib.gridspec.GridSpec
        grid of overall figure to plot
        performance plot at [3,0]

    """
    # get labels and time in correct format for heatmap plotting
    labels = case_inference_table.l.values.reshape(-1,1).T 
    t = case_inference_table.times.values/60 # convert to min

    #customize a binary colormap
    cmap_lbl = ListedColormap(['k','gray'])

    # build 1D axis colored by label at each timepoint
    first_ax = plt.gcf().axes[0]
    ax_lbl = plt.subplot(gridspec[3,0],sharex=first_ax)
    ax_lbl.pcolormesh(t,[0,1],labels,cmap=cmap_lbl, zorder=-10)
    ax_lbl.set_yticklabels([])
    ax_lbl.set_rasterization_zorder(-5)

    # ax_lbl.tick_params(axis='y',          # changes apply to the x-axis
    #                    which='both',      # both major and minor ticks are affected
    #                    bottom=False,      # ticks along the bottom edge are off
    #                    left=False)

    ax_lbl.set_xlabel('Time (min)')

    # build custom legend colored by depth of sedation represented in label plot
    ax_lbllgd = plt.subplot(gridspec[2:,1])
    ax_lbllgd.axis('off')


    # hackey custom legend
    ax_lbllgd.scatter(0.1,0.41,marker='s',color='k')
    ax_lbllgd.scatter(0.1,0.16,marker='s',color='gray')
    ax_lbllgd.text(0.3,0.35,'Unconscious', fontsize=8,color='k')
    ax_lbllgd.text(0.3,0.1,'Conscious', fontsize=8,color='k')
    ax_lbllgd.set_ylim([0,2])
    ax_lbllgd.set_xlim([0,2])
    

    return gridspec

def plt_drugs_or(data_dict, gridspec):
    """plotting function for relevant drugs given via bolus or infusion
    for OR patients
    
    Parameters
    ----------
    data_dict : dict
        expects keys 'infusion' or 'bolus' that point to tables with time, dose, and units of drug given
    gridspec : gridspec : matplotlib.gridspec.GridSpec
        grid of overall figure to plot
        drugs plot will be at [1,0]
    
    Returns
    -------
    matplotlib.gridspec.GridSpec
        grid of overall figure to plot
        drugs plot at [1,0]
    """
    
    # plot drug event times
    ax_infusion = plt.subplot(gridspec[1,0])
    ax_bolus = ax_infusion.twinx()
    ax_bolus_lgd = plt.subplot(gridspec[1,1])

    if 'infusion' in data_dict:
        infusion = data_dict['infusion']
        _infusion_plot(ax_infusion, infusion)
    if 'bolus' in data_dict:
        bolus = data_dict['bolus']
        _bolus_plot(bolus, ax_bolus, ax_bolus_lgd)
    return gridspec

def _infusion_plot(ax, infusion):
    """helper function for plotting OR infusion
    time on x axis is in min
    
    Parameters
    ----------
    ax : matplotlib.Axes
        axis on which to plot infusion
    infusion : numpy.ndarray
        array of infusions with drug name, start time, stop time (sec since eeg on), rate, units
    
    Returns
    -------
    None
        plot added to axis passed

    """
    # fix dimensions if 1d
    if len(infusion.shape) == 1: infusion = np.array([infusion])

    # get all unique drugs infused
    all_drugs = np.unique(infusion[:,0])
    non_p = all_drugs[np.where(all_drugs!='Propofol')[0]]
    if len(non_p) > 0: print("Infused drugs other than propofol are present.")
    inf_p_inds = np.where(infusion[:,0]=='Propofol')[0]

    # assemble time series - start w/ 0
    inf_p_ts = [infusion[inf_p_inds[0],1]]
    inf_p_doses = [0]
    for idx in inf_p_inds:
        inf_p_ts.append(infusion[idx, 1:3])
        inf_p_doses.append([infusion[idx][3], infusion[idx][3]])
    its = np.array(np.hstack(inf_p_ts), dtype=np.float)
    ids = np.array(np.hstack(inf_p_doses), dtype=np.float)
    # returns to 0
    its = np.hstack([its, [its[-1]]])
    ids = np.hstack([ids, [0]])

    ax.plot(its/60, ids, color='k')
    ax.set_ylabel('Propofol Infusion \n(mcg/kg/min)')
    ax.set_ylim([0, 300])

def _bolus_plot(bolus, ax, ax_lgd):
    """helper function to plot bolus info for OR 
    propofol boluses are plotted as stems with unique y axis on the right, can be ovarlayed with infusion plot
    all other boluses are givent a unique letter label on the x axis
    time on x axis is in minutes since EEG on
    only plot boluns for propofol. label all other drug boluses and list drug name and amt on the right.
    
    Parameters
    ----------
    bolus : numpy.ndarray
        table with bolus information for relevant drugs
        each row containes drug name, time in sec since eeg on, dose, unit
    ax : matplotlib.Axes
        axis to plot propofol boluses and times of non propofol boluses
    ax_lgd : matplotlib.Axes
        axis to plot legend for non propofol boluses
    """
    
    # plot specs
    prop_bolus_color='b'
    non_propofol_labels = list(string.ascii_uppercase)[::-1]

    ax.set_ylabel('Propofol Bolus \n(mg)', color=prop_bolus_color)    
    ax.tick_params(axis='y', labelcolor=prop_bolus_color)
    ax_lgd.axis('off')

    non_prop_drug_time = []
    non_prop_drug_legend_label = []

    # iterate through boluses
    for bol in bolus:
        drug=bol[0]
        time=float(bol[1])/60
        dose=np.round(float(bol[2]),3)
        units=bol[3]
    
        if 'propofol' in drug.lower():
            ax.stem([time],[dose],markerfmt='bo',linefmt=prop_bolus_color+':',
                           use_line_collection=True)
            assert units=='mg', 'units of propofol bolus not in mg'
        else:
            legend_label = f":{drug}, {dose}{units}"

            non_prop_drug_time.append(time)
            non_prop_drug_legend_label.append(legend_label)
    # sort by time becasue boluses are not being parsed in ordered time
    non_prop = pd.DataFrame({'t':non_prop_drug_time,'lbl':non_prop_drug_legend_label})
    non_prop.t = non_prop.t.astype(float)
    non_prop = non_prop.sort_values(by='t')
    non_prop['xtick_lbl'] = [non_propofol_labels.pop() for i in non_prop.index]


    # plot custom x ticks
    x_formatter = FixedFormatter(non_prop.xtick_lbl.values)
    x_locator = FixedLocator(non_prop.t.values)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_locator(x_locator)
    ax.tick_params(axis='x', labelsize=6)
    ax.set_ylim([0,100])

    # plot legend for non propofol boluses
    if np.any(non_prop): # check if list is empty
        labels = non_prop.xtick_lbl + non_prop.lbl
        ax_lgd.text(0,0,'\n'.join(labels),fontsize=8)

def _align_axes(fig):
    
    # get overall first time point and last time point across all 4 axes
    # All axes have units minutes since EEG on as x axis so the numbers are all directly comparable
    xmins=[]
    xmaxs=[]
    for ax_idx in AXES_TO_ALIGN:
        ax = fig.axes[ax_idx]
        xmins.append(ax.get_xlim()[0])
        xmaxs.append(ax.get_xlim()[1])
        
    xmin_overall = min(xmins)
    xmax_overall = max(xmaxs)

    # align relevant(AXES_TO_ALIGN) axes to overal min and max with 10min of padding
    for ax_idx in AXES_TO_ALIGN:
        ax = fig.axes[ax_idx]
        ax.set_xlim(xmin_overall, xmax_overall+10)
    
    return fig

def _load_volunteer_drug(case_id, DIR_RAW_HV):
    if case_id == '07':
        eeg_file = DIR_RAW_HV+'/eeganes'+case_id+'_detrend.mat'
    else:
        eeg_file = DIR_RAW_HV+'/eeganes'+case_id+'.mat'
    bhvr_file = DIR_RAW_HV+'/eeganes'+case_id+'_laplac250_ch36.mat'

    drug_times, drug_conc = nsrl.load_patrick2013mat(bhvr_file, eeg_file, 
                                                     hz=250, return_drug=True)[-2:]

    return drug_times, drug_conc

def _filter_inftbl(case_id,classifier_names,DIR_INFTBL):

    # read first inference table to make table with static columns (case_id, t, l)
    fp_inftbl_clf1 = DIR_INFTBL+classifier_names[0]+'.csv'
    assert os.path.isfile(fp_inftbl_clf1), f'inference table for {classifier_names[0]} not found in {DIR_INFTBL}'
    inftbl_multiclf = pd.read_csv(fp_inftbl_clf1).loc[:,['times','caseid','l','egq']]
    inftbl_multiclf.caseid = inftbl_multiclf.caseid.astype('category')
    # append predictions for each classifier in classifier_names
    for cname in classifier_names:
        fp_inftbl = DIR_INFTBL+cname+'.csv'
        preds = pd.read_csv(fp_inftbl).py
        inftbl_multiclf[cname+'_py'] = preds
    
    # fiter by case_id
    inftbl_multiclf = inftbl_multiclf.groupby('caseid').get_group(int(case_id))

    return inftbl_multiclf




def pca_case_plot(case_id):
    """
    Plots normed PCs and conscious or unconscious, and line between two.
    """
    #load multitaper info
    filepath = os.path.expanduser('~/Dropbox (Partners HealthCare)/HumanSignalsData/Subjects_Database/Curated_OR')
    data = cc.load_OR_todict(filepath, case_id)

    # norm the pcs
    dd_pc = data['PCs']
    dd_egq = data['egq']

    qts0 = np.where(dd_egq)[0][:15] # first 15 quality timepoints
    dd_pc0 = np.median(dd_pc[:, qts0], 1)

    pcdata = dd_pc.T - dd_pc0
    labels = data['l']

    # split data
    awake_ids = np.where(labels==1)[0]
    out_ids = np.where(labels==0)[0]

    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(pcdata[awake_ids, 0], pcdata[awake_ids, 1], 
            c='r', marker='o', ls='', label='Conscious')
    ax.plot(pcdata[out_ids, 0], pcdata[out_ids, 1], 
            c='b', marker='x', ls='', label='Unconscious')
    ax.plot([1000,1001], [1000,1001], 'k', label='LDA Decision Boundary')
    ax.set_xlim([-200,200])
    ax.set_ylim([-100,100])

    # plot decision function
    clf = pickle.load(open('Results/Abel/LDA/PCA_30snorm_2s.sav', 
                            'rb'))
    xx = np.linspace(-200, 200, 30)
    yy = np.linspace(-100, 100, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[0.5], 
           linestyles=['-'], zorder=10)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.legend()
    plt.tight_layout(**plo.layout_pad)
    return fig

def pca_hv_plot(case_id):
    """
    Plots normed PCs and conscious or unconscious, and line between two.
    """
    #load multitaper info
    filepath = os.path.expanduser('~/Dropbox (Partners HealthCare)/HumanSignalsData/Subjects_Database/Volunteer_Data')
    data = cc.load_volunteer_todict(filepath, case_id)

    # norm the pcs
    dd_pc = data['PCs']
    dd_egq = data['egq']

    qts0 = np.where(dd_egq)[0][:15] # first 15 quality timepoints
    dd_pc0 = np.median(dd_pc[:, qts0], 1)

    pcdata = dd_pc.T - dd_pc0
    labels = data['l']

    # split data
    awake_ids = np.where(labels==1)[0]
    out_ids = np.where(labels==0)[0]

    plt.figure()
    ax = plt.subplot()
    ax.plot(pcdata[awake_ids, 0], pcdata[awake_ids, 1], 
            c='r', marker='o', ls='', label='Conscious')
    ax.plot(pcdata[out_ids, 0], pcdata[out_ids, 1], 
            c='b', marker='x', ls='', label='Unconscious')
    ax.plot([1000,1001], [1000,1001], 'k', label='LDA Decision Boundary')
    ax.set_xlim([-200,200])
    ax.set_ylim([-100,100])

    # plot decision function
    clf = pickle.load(open('Results/Abel/LDA/PCA_30snorm_2s.sav' , 
                            'rb'))
    xx = np.linspace(-200, 200, 30)
    yy = np.linspace(-100, 100, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[0.5], 
           linestyles=['-'], zorder=10)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.legend()
    plt.tight_layout(**plo.layout_pad)\

def adjust_xlim(fig, xlim):
    """adjusts the xlim on all relevant axes
    
    Parameters
    ----------
    fig : matplotlib.figure
        [description]
    """    
    axes = fig.axes
    for ax in axes[::2]:
        ax.set_xlim(xlim)