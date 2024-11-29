import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.linalg
from sklearn import svm

import argparse
import importlib
import time
import datetime
import os
import sys
import copy
from collections import *

from utils import CC
from utils.multiple_testing import pBH, eBH, eBH_infty, evaluate
from utils.p2e import p2e
from utils.rej_functions import ebh_rej_function, ebh_infty_rej_function, ebh_union_rej_function
from utils.ci_sequence import get_alpha_cs, get_rho, hedged_cs, asy_cs, asy_log_cs 
from utils.generating_data import *

from Adadetect import AdaDetectDE, AdaDetectERM

from cc_utils.conformal import conformal


class SVMScoringFn:
    def __init__(self, nu, kernel, gamma):
        self.classifier = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit(self, xtrain, x_null_train):
        self.classifier.fit(x_null_train)

    def score_samples(self, X):
        return -1 * self.classifier.score_samples(X)

#HYPERPARAMETERS
seeds = np.arange(1,100)
alpha_fdr = 0.3
name = "Ada"
m = 500 #number of points in test
n = 1000 #number of points in the calibration
prop_outliers = 0.5 #proportion of outlier (pi) in the experiement

# details
amps = np.arange(2,4,0.1) # we create a range of parameters to visualize power curve
n_train = n # can specify training amount 

# The synthetic data
THETA = np.zeros(50) 

#The setting we're planning to investigate
p_theta = 6 
THETA[:p_theta,] = np.array([0.3, 0.3, 0.2, 0.2, 0.1, 0.1]) 
THETA = THETA.reshape((50,1))


### make directories
filepath = f'results/{name}/'
os.makedirs(filepath, exist_ok=1) 

# Wset
avg_power = np.zeros(len(amps))
iteration = 0

for seed in seeds:
    powers = []
    np.random.seed(seed)   # Wset should be same for all
    Wset = np.random.uniform(size=(50,50)) * 6 - 3

    n_outliers = int(np.ceil(m * prop_outliers))
    nonzero = np.array(range(m))[:n_outliers]    # the indices of the outliers
    nulls = np.array(range(m))[n_outliers:] 

    # Loop through through different strengths of a signal
    for amp in amps:
        # We simulate the data and then train SVM on it
        Xtrain = gen_data(Wset, n_train, 1)
        Xcalib = gen_data(Wset, n, 1)

        Xtest0 = gen_data_weighted(Wset, m-n_outliers, 1, THETA)    # inliers 
        Xtest1 = gen_data_weighted(Wset, n_outliers, amp, THETA)    # outliers

        Xtest = np.zeros((m, Xtest0.shape[1]))
        Xtest[nonzero,] = Xtest1
        Xtest[nulls,] = Xtest0

        #Initializing the Adadetect
        scoring_fn = svm.OneClassSVM(nu=0.004, kernel="rbf", gamma=0.1)
        adadetect = AdaDetectDE(scoring_fn=scoring_fn, f0_known=True)
        
        #We investigate the simple setting
        rej = adadetect.apply(Xtest, level=alpha_fdr, xnull=Xtrain)
        print(rej)

        # adadetect output baseline
        powers.append(evaluate(rej, nonzero)['power'])
    
    iteration += 1 
    avg_power = np.add(avg_power, powers)

    print(f'Simulation {iteration} out of {len(seeds)} over')
    print(f"Computed powers {powers}")
    print(f"Sum of powers {avg_power}")

# Saving the power curve as a file
graph_name = 'Adadetect'
file_path = f"results/{name}/{graph_name}.png"
os.makedirs(filepath, exist_ok=1)    # name should be the results filepath


plt.figure(figsize=(8, 6))
plt.plot(amps, avg_power/len(seeds), label='Power Curve', linewidth=2)
plt.title('Power Curve', fontsize=16)
plt.xlabel('Amplitude', fontsize=14)
plt.ylabel('Power', fontsize=14)
plt.grid(alpha=0.5)
plt.legend(fontsize=12)
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()


#Section 6.2: Implementation of p-values  (V_i \geq V_{n+j} is either the plit conformal or full-conformal function)
# Implementation - split conformal it's good/ full conformal it's bad.
# Use the Adadetect from the paper.
# Full conformal p-values, split 
#PALMRT paper - review it. 