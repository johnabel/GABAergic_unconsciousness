#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12-6-2019 jha

This file performs crossvalidation for all models. Currently does not work for CNN features (not yet incorporated).
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


# all cases are propofol cases

# fig6 
fig6a = viz.timeseries_or('697')
viz.adjust_xlim(fig6a, [-5,45])
fig6b = viz.timeseries_or('154')
viz.adjust_xlim(fig6b, [-1,38])

#s2
figs1 = viz.timeseries_or('455')
viz.adjust_xlim(figs1, [-12,41])