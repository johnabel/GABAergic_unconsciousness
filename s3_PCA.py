#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10-7-19
This file gets all PCs and the LD.
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
from LocalResources import Visualization as viz

# instationtion of a classifier object
clf = cc.ConsciousnessClassifier(name='linear')

# get the directories
DIR_TRAIN = 'Data/Volunteer/' 
DIR_VALIDATION = 'Data/Volunteer/' 
DIR_MODEL = 'Results/Trained Models/'

train_names = ['03', '02', '04', '05', '07', '08', '09']
validation_names = ['10']

# now load the dataset
clf.load_dataset(DIR_TRAIN, train_names, dsetname='training', subjects='volunteer')
clf.load_dataset(DIR_VALIDATION, validation_names, dsetname='case10', subjects='volunteer')

# add and fit the three classifiers
models = ["Sdb", "PCA", "LDA+HMM2"]
ctypes = ['lr', 'lr', 'lr']
ttypes = ['standard', 'standard', 'hmm2']
ftypes = ['Sdb', 'PCA', 'LDA']

for mi, mn in enumerate(models):
    clf.add_model(ctype=ctypes[mi], ftype=ftypes[mi], ttype=ttypes[mi], mname=mn)
    clf.train_model(mn, "training", which_features=['Sdb','bands','PCA','LDA'])
    clf.validate_model(mn, "case10")

# get the contents for training and the application subject
training_set = clf.datasets['training']
subj = clf.datasets['case10']

# a couple plots of PCA
plo.PlotOptions(ticks='in')

plt.figure(figsize=(4.,6.5))
gs = gridspec.GridSpec(5,3, height_ratios=(1,1,1,1,1), width_ratios=(10,10,1))

mx = plt.subplot(gs[0,:2])
cb = mx.pcolormesh(subj['times']/60, subj['fs'], subj['Sdb'], vmin=-10, vmax=30, 
                   cmap='magma', zorder=-10)
loc, roc = subj['times'][:-1][np.abs(np.diff(subj['labels']))>0.5]
mx.axvline(loc/60, c='white')
mx.axvline(roc/60, c='white')
mx.set_ylabel('Freq (Hz)')
mx.set_xlim([0,155])
mx.set_rasterization_zorder(-5)
cbx = plt.subplot(gs[0,2])
plt.colorbar(cb, cax=cbx)
cbx.set_ylabel('Power (dB)')
mx.set_xlabel('Time (min)')

# featurization results
_, case10_feats = clf.featurize(training_set['df'], training_set['fs'], subj['df'], which_features=['Sdb','bands','PCA','LDA'])
pca, lda = clf.pca_lda

ax = plt.subplot(gs[1,0])
ax.plot(np.arange(1,11), pca.explained_variance_ratio_[:10], 'k')
ax.set_ylim([0,0.7])
ax.set_xlim([0,10])
ax.set_xlabel('PC Number')
ax.set_ylabel('Explained Var')

bx = plt.subplot(gs[1,1])
plt.axhline(0, c='0.5')
bx.plot(subj['fs'], pca.components_[0], label='PC1', c='i')
bx.plot(subj['fs'], pca.components_[1], label='PC2', c='h')
bx.plot(subj['fs'], pca.components_[2], label='PC3', c='f')
bx.set_ylim([-0.25, 0.25])
bx.set_xlim([0,50])
bx.set_ylabel('Eigenval')
bx.set_xlabel('Freq (Hz)')
plt.legend(loc=0)

cx = plt.subplot(gs[2,:2])
feats = case10_feats['PCA']
cx.plot(feats['times']/60, feats[0], label='PC1', c='i', alpha=0.75)
cx.plot(feats['times']/60, feats[1], label='PC2', c='h', alpha=0.75)
cx.plot(subj['times']/60, feats[2], label='PC3', c='f', alpha=0.75)
cx.set_xlabel('Time (min)')
cx.set_ylabel('PC Score')
cx.set_xlim([0,155])
cx.set_ylim([-75, 250])
plt.legend(loc=2)
cx.set_xlabel('Time (min)')
cx.set_ylabel('PC Score')

dx = plt.subplot(gs[3,1])
plt.axhline(0, c='0.5')
dx.plot(subj['fs'], lda.coef_[0], label='LD', c='k', alpha=0.75)
dx.set_xlabel('LD Coefficient')
dx.set_xlabel('Freq (Hz)')
dx.set_ylim([-0.25, 0.25])
dx.set_xlim([0,50])
plt.legend(loc=2)
dx.set_ylabel('Linear Discriminant')
dx.set_xlabel('Freq (Hz)')

ex = plt.subplot(gs[4,:2])
feats = case10_feats['LDA']
ex.plot(subj['times']/60, feats[0], c='k', alpha=0.75)
ex.set_xlabel('Time (min)')
ex.set_ylabel('LD Score')
ex.set_xlim([0,155])
ex.set_ylim([-4, 4])

plt.tight_layout(**plo.layout_pad)