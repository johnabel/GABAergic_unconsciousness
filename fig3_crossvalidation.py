#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12-6-2019 jha

This file performs crossvalidation for all models. CNN features are available on-demand, but as-is we don't include them due to the large file size. In all cases here the use of the CNN features is commented out.

In the end, this trains the selected three classification models.
"""

# python numerics
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, auc

# plotting
import matplotlib.pyplot as plt
from matplotlib import gridspec
import prettyplotlib as ppl
from prettyplotlib import brewer2mpl
import seaborn as sns

# local
from LocalResources import Utilities as util
from LocalResources import NSRL_resources as nsrl
from LocalResources import PlotOptions as plo
from LocalResources import ConsciousnessClassifier as cc

# instationtion of a classifier
clf = cc.ConsciousnessClassifier(name='linear')

# get the directories for train, int, ext

# crossvalidation
train_dir = 'Data/Volunteer/'
train_names = ['03', '02', '04', '05', '07', '08', '09']

validation_dir = train_dir
validation_names = ['10', '13', '15']

# now load the dataset
clf.load_dataset(train_dir, train_names, dsetname='training', subjects='volunteer')
clf.load_dataset(validation_dir, validation_names, dsetname='validation', subjects='volunteer')

# add and fit all classifiers
ctypes = ['lr']
ttypes = ['standard', 'hmm2', 'hmmfree']
ftypes = ['Sdb', 'bands', 'PCA', 'LDA']#, 'CNN']

cval_aucs = []
mnames = []

for ct in ctypes:
    for tt in ttypes:
        for ft in ftypes:
            mname = f"{ct}_{tt}_{ft}"
            print(mname+": "),
            clf.add_model(ctype=ct, ftype=ft, ttype=tt, mname=mname)
            cval_out = clf.crossvalidation(mname, 'training', features=ftypes)
            print(np.round(np.mean(cval_out),3))
            
            # attach output
            cval_aucs.append(cval_out)
            mnames.append(mname)

# save table at the end
df = pd.DataFrame(data=np.array(cval_aucs).T, columns=mnames)
df.to_csv('Results/crossval.csv')

# plotting
xs = np.arange(15)
cs = sns.color_palette("husl", n_colors=15)

fig = plt.figure(figsize=(6.8,2.6))
gs = gridspec.GridSpec(1,2, width_ratios=(2,1))
ax=plt.subplot(gs[0,0])
for ci, cva in enumerate(cval_aucs):
    if ci%2==0:
        ax.fill_between([xs[ci]-0.5, xs[ci]+0.5], [2,2], [0,0], color='0.9')
    ax.plot(xs[ci]+np.arange(7)*0.05-0.175, cva, color=cs[ci], ls='', marker='o', markersize=2)
    ax.plot([xs[ci]-0.1, xs[ci]+0.1], [np.mean(cva), np.mean(cva)], 'k')
ax.set_xlabel('Model')
ax.set_ylabel("Crossvalidation AUC")
ax.set_ylim([0,1.02])
ax.set_xticks(xs)
ax.set_xlim([-0.5,14.5])
ax.set_xticklabels(['SDB', 'BWP', "PCA", "LDA"]*3)#, "CNN"]*3)

# heatmap
# get human-readable names
# mnames_hr = ['SDB', 'BWP', "PCA", "LDA", "CNN"] + \
#              [aa + '+HMM2' for aa in ['SDB', 'BWP', "PCA", "LDA", "CNN"]] + \
#              [aa + '+HMM6' for aa in ['SDB', 'BWP', "PCA", "LDA", "CNN"]]
mnames_hr = ['SDB', 'BWP', "PCA", "LDA", "CNN"] + \
             [aa + '+HMM2' for aa in ['SDB', 'BWP', "PCA", "LDA"]] + \
             [aa + '+HMM6' for aa in ['SDB', 'BWP', "PCA", "LDA"]]

# sort from highest to lowest mean AUC
mean_aucs = np.array([np.mean(aa) for aa in cval_aucs])
aucs_resort = [cv for _,cv in sorted(zip(mean_aucs, cval_aucs))][::-1]
names_resort = [nn for _,nn in sorted(zip(mean_aucs, mnames_hr))][::-1]
means_sorted = np.sort(mean_aucs)[::-1]

bx = plt.subplot(gs[0,1])
diffs = np.zeros([len(names_resort), len(names_resort)])
for m1i, mn1 in enumerate(names_resort):
    for m2i, mn2 in enumerate(names_resort):
        if m2i !=m1i:
            diffs[m1i, m2i] = means_sorted[m2i]- means_sorted[m1i]
red_purple = brewer2mpl.get_map('RdBu', 'Diverging', 9).mpl_colormap
ppl.pcolormesh(fig, bx, diffs.T, cmap=red_purple, 
               xticklabels = names_resort,
               yticklabels = names_resort, vmax=0.6, vmin=-0.6)
bx.xaxis.tick_top()
plt.xticks(rotation=90)
for tic in bx.xaxis.get_major_ticks():
    tic.tick1On = tic.tick2On = False
plt.gca().invert_yaxis()
plt.tight_layout(**plo.layout_pad)
#plt.savefig('Results/fig3_crossval.svg')

# finally, train the models that we are keeping and save the outputs
# saving is currently commented out so as to not overwrite the models actually used in the MS
human_readable_names = ['Sdb', 'PCA', 'LDA+HMM2']
for mi, mname in enumerate(['lr_standard_Sdb', 'lr_standard_PCA', 'lr_hmm2_LDA']):
    clf.train_model(mname, 'training', 
                    which_features=['Sdb','bands','PCA','LDA'])
    #clf.save_model('Results/'+human_readable_names[mi]+'.p', mname)