"""
original: Benyamin Meschede-Krasa 12-13-2019 
last modified: John Abel 7-21-2020

script for OR validations of 3 selected models
expects models to already be trained
"""

# python numerics
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, auc
import yaml

# plotting
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from LocalResources.tol_colors import tol_cset

# local
from LocalResources import Utilities as util
from LocalResources import NSRL_resources as nsrl
from LocalResources import PlotOptions as plo
from LocalResources import ConsciousnessClassifier as cc


# Parameters
DIR_YML = 'Data/OR/rx_sorted_case_ids.yml'
DIR_EXTERNAL_VALIDATION = 'Data/OR/'
DIR_MODEL = 'Results/Trained Models/'
DIR_INF_TBL_OUT = 'Results/OR Inference Tables/'


# Preconditions
assert os.path.exists(DIR_MODEL)
assert os.path.exists(DIR_EXTERNAL_VALIDATION)
assert os.path.exists(DIR_INF_TBL_OUT)
assert os.path.exists(DIR_YML)

# instationtion of a classifier object
clf = cc.ConsciousnessClassifier(name='linear')

# now load the dataset
with open(DIR_YML,'r') as fi:
    sorted_cases = yaml.load(fi,Loader=yaml.FullLoader)

case_ids_propofol = sorted_cases['pure_propofol']
case_ids_sevoflurane = sorted_cases['mixed']+sorted_cases['pure_sevo']

clf.load_dataset(DIR_EXTERNAL_VALIDATION, case_ids_propofol, 
                 dsetname='propofol', subjects='OR')
clf.load_dataset(DIR_EXTERNAL_VALIDATION, case_ids_sevoflurane, 
                 dsetname='sevoflurane', subjects='OR')

# load the three fitted classifiers
models = ["Sdb", "PCA", "LDA+HMM2"]
ctypes = ['lr', 'lr', 'lr']
ttypes = ['standard', 'standard', 'hmm2']
ftypes = ['Sdb', 'PCA', 'LDA']

for mi, mn in enumerate(models):
    clf.load_model(DIR_MODEL+mn+'.p', mn)
    clf.validate_model(mn, "propofol")
    clf.validate_model(mn, "sevoflurane")
    propofol_tab = clf.models[mn].val_result["propofol"]
    sevo_tab = clf.models[mn].val_result["sevoflurane"]
    inf_tab = pd.concat([propofol_tab, sevo_tab])
    inf_tab.to_csv(DIR_INF_TBL_OUT+mn+'.csv')



# now let's get the AUC results

# propofol
fpr1p, tpr1p, thr1p, auc1p, thr_opt_1p = clf.roc_auc("Sdb", "propofol")
fpr2p, tpr2p, thr2p, auc2p, thr_opt_2p = clf.roc_auc("PCA", "propofol")
fpr3p, tpr3p, thr3p, auc3p, thr_opt_3p = clf.roc_auc("LDA+HMM2", "propofol")

# sevoflurane
fpr1m, tpr1m, thr1m, auc1m, thr_opt_1m = clf.roc_auc("Sdb", "sevoflurane")
fpr2m, tpr2m, thr2m, auc2m, thr_opt_2m = clf.roc_auc("PCA", "sevoflurane")
fpr3m, tpr3m, thr3m, auc3m, thr_opt_3m = clf.roc_auc("LDA+HMM2", "sevoflurane")


# get classifier AUC, thr, ACC and optimal ACC - propofol
auc_sdb_p = []
auc_pca_p = []
auc_lda_p = []
thr_sdb_p = []
thr_pca_p = []
thr_lda_p = []
acc_sdb_p = []
acc_pca_p = []
acc_lda_p = []
acc_sdb_popt = []
acc_pca_popt = []
acc_lda_popt = []
for case in case_ids_propofol:
    auc_sdb_p.append(clf.roc_auc("Sdb", "propofol", cases=[case])[3])
    auc_pca_p.append(clf.roc_auc("PCA", "propofol", cases=[case])[3])
    auc_lda_p.append(clf.roc_auc("LDA+HMM2", "propofol", cases=[case])[3])
    thr_sdb_p.append(clf.roc_auc("Sdb", "propofol", cases=[case])[4])
    thr_pca_p.append(clf.roc_auc("PCA", "propofol", cases=[case])[4])
    thr_lda_p.append(clf.roc_auc("LDA+HMM2", "propofol", cases=[case])[4])
    acc_sdb_p.append(clf.acc("Sdb", "propofol", cases=[case]))
    acc_pca_p.append(clf.acc("PCA", "propofol", cases=[case]))
    acc_lda_p.append(clf.acc("LDA+HMM2", "propofol", cases=[case]))
    acc_sdb_popt.append(clf.acc("Sdb","propofol", cases=[case], 
                        thr=thr_sdb_p[-1]))
    acc_pca_popt.append(clf.acc("PCA","propofol", cases=[case], 
                        thr=thr_pca_p[-1]))
    acc_lda_popt.append(clf.acc("LDA+HMM2", "propofol", cases=[case], 
                        thr=thr_lda_p[-1]))

# get classifier AUC, thr, ACC and optimal ACC - propofol
auc_sdb_m = []
auc_pca_m = []
auc_lda_m = []
thr_sdb_m = []
thr_pca_m = []
thr_lda_m = []
acc_sdb_m = []
acc_pca_m = []
acc_lda_m = []
acc_sdb_mopt = []
acc_pca_mopt = []
acc_lda_mopt = []
for case in case_ids_sevoflurane:
    auc_sdb_m.append(clf.roc_auc("Sdb", "sevoflurane", cases=[case])[3])
    auc_pca_m.append(clf.roc_auc("PCA", "sevoflurane", cases=[case])[3])
    auc_lda_m.append(clf.roc_auc("LDA+HMM2", "sevoflurane", cases=[case])[3])
    thr_sdb_m.append(clf.roc_auc("Sdb", "sevoflurane", cases=[case])[4])
    thr_pca_m.append(clf.roc_auc("PCA", "sevoflurane", cases=[case])[4])
    thr_lda_m.append(clf.roc_auc("LDA+HMM2", "sevoflurane", cases=[case])[4])
    acc_sdb_m.append(clf.acc("Sdb", "sevoflurane", cases=[case]))
    acc_pca_m.append(clf.acc("PCA", "sevoflurane", cases=[case]))
    acc_lda_m.append(clf.acc("LDA+HMM2", "sevoflurane", cases=[case]))
    acc_sdb_mopt.append(clf.acc("Sdb","sevoflurane", cases=[case], 
                        thr=thr_sdb_m[-1]))
    acc_pca_mopt.append(clf.acc("PCA","sevoflurane", cases=[case], 
                        thr=thr_pca_m[-1]))
    acc_lda_mopt.append(clf.acc("LDA+HMM2", "sevoflurane", cases=[case], 
                        thr=thr_lda_m[-1]))


# get human-readable names, get colors used in model selection boxplot
mnames_hr = ["SDB", "PCA", "LDA+HMM2"]
cset = tol_cset('bright')
cs = cset[:3]


# plotting - propofol
plt.figure(figsize=(3.5,3))
gs = gridspec.GridSpec(2,2)

bx=plt.subplot(gs[0,0])
xs = np.arange(3)
for ai, auc in enumerate([auc_sdb_p, auc_pca_p, auc_lda_p]):
    bx.plot(xs[ai]+np.arange(27)*0.02-0.27, auc, color=cs[ai], ls='', marker='o', markersize=2)
    bx.plot([xs[ai]-0.1, xs[ai]+0.1], [np.median(auc), np.median(auc)], 'k')
# sns.boxplot(data=[auc_sdb, auc_pca, auc_lda], ax=bx, fliersize=1.5, palette=cs)
bx.set_ylabel("Casewise AUC")
bx.set_ylim([0, 1.02])
bx.set_xticks([0, 1, 2])
bx.set_xticklabels(['SDB', 'PCA', "LDA\n+HMM2"])
print(f"Propofol AUC: {[np.median(auc) for auc in [auc_sdb_p, auc_pca_p, auc_lda_p]]}")

cx=plt.subplot(gs[0,1])
xs = np.arange(27)
for ai, acc in enumerate([acc_sdb_p, acc_pca_p, acc_lda_p]):
    cx.plot(xs[ai]+np.arange(27)*0.02-0.27, acc, color=cs[ai], ls='', marker='o', markersize=2)
    cx.plot([xs[ai]-0.1, xs[ai]+0.1], [np.median(acc), np.median(acc)], 'k')
# sns.boxplot(data=[auc_sdb, auc_pca, auc_lda], ax=bx, fliersize=1.5, palette=cs)
cx.set_ylabel("ACC, Threshold=0.5")
cx.set_ylim([0, 1.02])
cx.set_xticks([0, 1, 2])
cx.set_xticklabels(['SDB', 'PCA', "LDA\n+HMM2"])
# case 455 is the bad one
print(f"Propofol ACC: {[np.median(acc) for acc in [acc_sdb_p, acc_pca_p, acc_lda_p]]}")

dx=plt.subplot(gs[1,0])
xs = np.arange(27)
for ai, acc in enumerate([acc_sdb_popt, acc_pca_popt, acc_lda_popt]):
    dx.plot(xs[ai]+np.arange(27)*0.02-0.27, acc, color=cs[ai], ls='', marker='o', markersize=2)
    dx.plot([xs[ai]-0.1, xs[ai]+0.1], [np.median(acc), np.median(acc)], 'k')
# sns.boxplot(data=[auc_sdb, auc_pca, auc_lda], ax=bx, fliersize=1.5, palette=cs)
#dx.set_xlabel('OR Cohort')
dx.set_ylabel("ACC, Optimal Thresholds")
dx.set_ylim([0, 1.02])
dx.set_xticks([0, 1, 2])
dx.set_xticklabels(['SDB', 'PCA', "LDA\n+HMM2"])
# case 455 is the bad one
print(f"Propofol optACC: {[np.median(acc) for acc in [acc_sdb_popt, acc_pca_popt, acc_lda_popt]]}")

ex=plt.subplot(gs[1,1])
xs = np.arange(27)
for ai, acc in enumerate([thr_sdb_p, thr_pca_p, thr_lda_p]):
    ex.plot(xs[ai]+np.arange(27)*0.02-0.27, acc, color=cs[ai], ls='', marker='o', markersize=2)
    ex.plot([xs[ai]-0.1, xs[ai]+0.1], [np.median(acc), np.median(acc)], 'k')
# sns.boxplot(data=[auc_sdb, auc_pca, auc_lda], ax=bx, fliersize=1.5, palette=cs)
#ex.set_xlabel('OR Cohort')
ex.set_ylabel("Optimal Threshold")
ex.set_ylim([0, 1.02])
ex.set_xticks([0, 1, 2])
ex.set_xticklabels(['SDB', 'PCA', "LDA\n+HMM2"])
plt.tight_layout(**plo.layout_pad)
# case 455 is the bad one
print(f"Propofol thr opt: {[np.median(acc) for acc in [thr_sdb_p, thr_pca_p, thr_lda_p]]}")





# plotting - sevoflurane
plt.figure(figsize=(3.5,3))
gs = gridspec.GridSpec(2,2)

bx=plt.subplot(gs[0,0])
xs = np.arange(3)
for ai, auc in enumerate([auc_sdb_m, auc_pca_m, auc_lda_m]):
    bx.plot(xs[ai]+np.arange(17)*0.02-0.17, auc, color=cs[ai], ls='', marker='o', markersize=2)
    bx.plot([xs[ai]-0.1, xs[ai]+0.1], [np.median(auc), np.median(auc)], 'k')
# sns.boxplot(data=[auc_sdb, auc_pca, auc_lda], ax=bx, fliersize=1.5, palette=cs)
bx.set_ylabel("Casewise AUC")
bx.set_ylim([0, 1.02])
bx.set_xticks([0, 1, 2])
bx.set_xticklabels(['SDB', 'PCA', "LDA\n+HMM2"])
print(f"sevoflurane  AUC: {[np.median(auc) for auc in [auc_sdb_m, auc_pca_m, auc_lda_m]]}")

cx=plt.subplot(gs[0,1])
xs = np.arange(27)
for ai, acc in enumerate([acc_sdb_m, acc_pca_m, acc_lda_m]):
    cx.plot(xs[ai]+np.arange(17)*0.02-0.17, acc, color=cs[ai], ls='', marker='o', markersize=2)
    cx.plot([xs[ai]-0.1, xs[ai]+0.1], [np.median(acc), np.median(acc)], 'k')
# sns.boxplot(data=[auc_sdb, auc_pca, auc_lda], ax=bx, fliersize=1.5, palette=cs)
cx.set_ylabel("ACC, Threshold=0.5")
cx.set_ylim([0, 1.02])
cx.set_xticks([0, 1, 2])
cx.set_xticklabels(['SDB', 'PCA', "LDA\n+HMM2"])
print(f"sevoflurane  ACC: {[np.median(acc) for acc in [acc_sdb_m, acc_pca_m, acc_lda_m]]}")

dx=plt.subplot(gs[1,0])
xs = np.arange(27)
for ai, acc in enumerate([acc_sdb_mopt, acc_pca_mopt, acc_lda_mopt]):
    dx.plot(xs[ai]+np.arange(17)*0.02-0.17, acc, color=cs[ai], ls='', marker='o', markersize=2)
    dx.plot([xs[ai]-0.1, xs[ai]+0.1], [np.median(acc), np.median(acc)], 'k')
dx.set_ylabel("ACC, Optimal Thresholds")
dx.set_ylim([0, 1.02])
dx.set_xticks([0, 1, 2])
dx.set_xticklabels(['SDB', 'PCA', "LDA\n+HMM2"])
print(f"sevoflurane  optACC: {[np.median(acc) for acc in [acc_sdb_mopt, acc_pca_mopt, acc_lda_mopt]]}")

ex=plt.subplot(gs[1,1])
xs = np.arange(17)
for ai, acc in enumerate([thr_sdb_m, thr_pca_m, thr_lda_m]):
    ex.plot(xs[ai]+np.arange(17)*0.02-0.17, acc, color=cs[ai], ls='', marker='o', markersize=2)
    ex.plot([xs[ai]-0.1, xs[ai]+0.1], [np.median(acc), np.median(acc)], 'k')
ex.set_ylabel("Optimal Threshold")
ex.set_ylim([0, 1.02])
ex.set_xticks([0, 1, 2])
ex.set_xticklabels(['SDB', 'PCA', "LDA\n+HMM2"])
plt.tight_layout(**plo.layout_pad)
print(f"sevoflurane  thr opt: {[np.median(acc) for acc in [thr_sdb_m, thr_pca_m, thr_lda_m]]}")




