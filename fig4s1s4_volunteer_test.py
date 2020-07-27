#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12-6-2019 jha

This file performs validation using the three held-out subjects.
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
from LocalResources.tol_colors import tol_cset

# local
from LocalResources import Utilities as util
from LocalResources import NSRL_resources as nsrl
from LocalResources import PlotOptions as plo
from LocalResources import ConsciousnessClassifier as cc
from LocalResources import Visualization as viz



# instationtion of a classifier object
clf = cc.ConsciousnessClassifier(name='linear')

# get the directories for train, int, ext
DIR_VALIDATION = 'Data/Volunteer/' 
DIR_MODEL = 'Results/Trained Models/'
NAMES_VALIDATION = ['10', '13', '15']

# now load the dataset
clf.load_dataset(DIR_VALIDATION, NAMES_VALIDATION, dsetname='internal', subjects='volunteer')

# add and fit the three classifiers
models = ["Sdb", "PCA", "LDA+HMM2"]
ctypes = ['lr', 'lr', 'lr']
ttypes = ['standard', 'standard', 'hmm2']
ftypes = ['Sdb', 'PCA', 'LDA']

for mi, mn in enumerate(models):
    clf.load_model(DIR_MODEL+mn+'.p', mn)
    clf.validate_model(mn, "internal")
    clf.save_inference_table('Results/Volunteer Inference Tables/', mn, "internal")

# now let's get the AUC results
fpr1, tpr1, thr1, auc1, _ = clf.roc_auc("Sdb", "internal")
fpr2, tpr2, thr2, auc2, _ = clf.roc_auc("PCA", "internal")
fpr3, tpr3, thr3, auc3, _ = clf.roc_auc("LDA+HMM2", "internal")

# get classifier case-wise AUC
auc_sdb = []
auc_pca = []
auc_lda = []
for case in NAMES_VALIDATION:
    auc_sdb.append(clf.roc_auc("Sdb", "internal", cases=[case])[3])
    auc_pca.append(clf.roc_auc("PCA", "internal", cases=[case])[3])
    auc_lda.append(clf.roc_auc("LDA+HMM2", "internal", cases=[case])[3])

# get classifier case-wise accuracy
acc_sdb = []
acc_pca = []
acc_lda = []
for case in NAMES_VALIDATION:
    acc_sdb.append(clf.acc("Sdb", "internal", cases=[case]))
    acc_pca.append(clf.acc("PCA", "internal", cases=[case]))
    acc_lda.append(clf.acc("LDA+HMM2", "internal", cases=[case]))

# get human-readable names, get colors used in model selection boxplot
mnames_hr = ["SDB", "PCA", "LDA+HMM2"]
cset = tol_cset('bright')
cs = cset[:3]


# plotting
plt.figure(figsize=(3.5,1.5))
gs = gridspec.GridSpec(1,2)
ax = plt.subplot(gs[0,0])

bx=plt.subplot(gs[0,0])
xs = np.arange(3)
for ai, auc in enumerate([auc_sdb, auc_pca, auc_lda]):
    bx.plot(xs[ai]+np.arange(3)*0.1-0.1, auc, color=cs[ai], ls='', marker='o', markersize=2)
    bx.plot([xs[ai]-0.1, xs[ai]+0.1], [np.median(auc), np.median(auc)], 'k')
# sns.boxplot(data=[auc_sdb, auc_pca, auc_lda], ax=bx, fliersize=1.5, palette=cs)
bx.set_xlabel('Model')
bx.set_ylabel("Casewise AUC")
bx.set_ylim([0, 1.02])
bx.set_xticks([0, 1, 2])
bx.set_xticklabels(['SDB', 'PCA', "LDA\n+HMM2"])

cx=plt.subplot(gs[0,1])
xs = np.arange(3)
for ai, acc in enumerate([acc_sdb, acc_pca, acc_lda]):
    cx.plot(xs[ai]+np.arange(3)*0.1-0.1, acc, color=cs[ai], ls='', marker='o', markersize=2)
    cx.plot([xs[ai]-0.1, xs[ai]+0.1], [np.median(acc), np.median(acc)], 'k')
# sns.boxplot(data=[auc_sdb, auc_pca, auc_lda], ax=bx, fliersize=1.5, palette=cs)
cx.set_ylabel("ACC, Threshold = 0.5")
cx.set_ylim([0, 1.02])
cx.set_xticks([0, 1, 2])
cx.set_xticklabels(['SDB', 'PCA', "LDA\n+HMM2"])
plt.tight_layout(**plo.layout_pad)
print(f"AUC: {[np.median(auc) for auc in [auc_sdb, auc_pca, auc_lda]]}")
print(f"ACC: {[np.median(acc) for acc in [acc_sdb, acc_pca, acc_lda]]}")



# and now for the bottom part
fig = viz.timeseries_hv('15')
axes = plt.gcf().get_axes()
for ax in [axes[0], axes[2]]:
    ax.set_xlim([0,155])

# and the supplrementary s5
# and now for the bottom part
fig = viz.timeseries_hv('13')
axes = plt.gcf().get_axes()
for ax in [axes[0], axes[2]]:
    ax.set_xlim([0,155])


