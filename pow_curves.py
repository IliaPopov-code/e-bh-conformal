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

from cc_utils.conformal import conformal


parser = argparse.ArgumentParser(description='Parser for siulations of e-BH-CC.')

# experiment config
parser.add_argument('--name', type=str, default="Exp", help='the name associated with the experiment.') #I changed from "required = True"
parser.add_argument('--n_exp', type=int, default=100, help='number of experiments (seeds) to run.')
parser.add_argument('--start_seed', type=int, default=0, help='the seed to with which to start when running the total experiment.')
parser.add_argument('--end_seed', type=int, default=None, help='the last seed in the total experiment.')

# experiment details
parser.add_argument('--m', type=int, default=500, help='number of observations in the test dataset.')
parser.add_argument('--n', type=int, default=1000, help='number of observations in the calibration dataset.') 
parser.add_argument('--n_train', type=int, default=None, help='number of observations in the training dataset.') 
parser.add_argument('--prop_outliers', type=float, default=0.1, help='frequency of outliers in the test dataset.') 

# amplitude for the experiment
parser.add_argument('--amp', type=float, default = 2.0, help='the signal for the true alternatives.')  #I changed from "required = True"

args = parser.parse_args()
print(args)

### make directories
filepath = f'results/{args.name}/'
os.makedirs(filepath, exist_ok=1) 

THETA = np.zeros(50) 

#The setting we're planning to investigate
p_theta = 6 
THETA[:p_theta,] = np.array([0.3, 0.3, 0.2, 0.2, 0.1, 0.1]) 
THETA = THETA.reshape((50,1))

# details
m = args.m        # number of hypotheses
n = args.n        # length of data matrix
amps = np.arange(2,4, 0.25) # we create a range of parameters to visualize power curve
powers = []

n_train = n if args.n_train == None else args.n_train    # can specify training amount 

prop_outliers = args.prop_outliers

# Wset
np.random.seed(42)   # Wset should be same for all
Wset = np.random.uniform(size=(50,50)) * 6 - 3

n_outliers = int(np.ceil(m * prop_outliers))
nonzero = np.array(range(m))[:n_outliers]    # the indices of the outliers
nulls = np.array(range(m))[n_outliers:] 

for amp in amps:
    # We simulate the data and then train SVM on it
    Xtrain = gen_data(Wset, n_train, 1)
    Xcalib = gen_data(Wset, n, 1)

    Xtest0 = gen_data_weighted(Wset, m-n_outliers, 1, THETA)    # inliers 
    Xtest1 = gen_data_weighted(Wset, n_outliers, amp, THETA)    # outliers

    Xtest = np.zeros((m, Xtest0.shape[1]))
    Xtest[nonzero,] = Xtest1
    Xtest[nulls,] = Xtest0

    # scoring function; training phase
    classifier = svm.OneClassSVM(nu=0.004, kernel="rbf", gamma=0.1)
    classifier.fit(Xtrain)

    print(Xtrain.shape, Xtest.shape)

    # compute calibration scores
    # scores should be that larger values <=> more likely outliers
    calib_weights = gen_weight(Xcalib, THETA).flatten() 
    calib_scores = -1 * classifier.score_samples(Xcalib).flatten() 
    cal_set = []
    for i in range(len(calib_weights)):
        cal_set.append( [calib_weights[i], calib_scores[i]] )

    # compute test scores 
    test_weights = gen_weight(Xtest, THETA).flatten()
    test_scores = -1 * classifier.score_samples(Xtest).flatten() #We could extend it to weighted
    test_set = []
    for i in range(len(test_weights)):
        test_set.append( [test_weights[i], test_scores[i]] )
        
    print(f"max cw: {max(calib_weights)} - min cw: {min(calib_weights)}")
    print(f"max tw: {max(test_weights)} - min tw: {min(test_weights)}")

    #We investigate the simple setting
    alpha_fdr = 0.1
    evalue_setting = conformal(alpha_conf=alpha_fdr)
    e_original = evalue_setting.e_function(cal_set, test_set)

    # eBH baseline
    rej = eBH(e_original, alpha_fdr)
    powers.append(evaluate(rej, nonzero)['power'])


# Saving the power curve as a file
graph_name = 'e-BH'
file_path = f"results/{args.name}/{graph_name}.png"
os.makedirs(filepath, exist_ok=1)    # name should be the results filepath


plt.figure(figsize=(8, 6))
plt.plot(amps, powers, label='Power Curve', linewidth=2)
plt.title('Power Curve', fontsize=16)
plt.xlabel('Amplitude', fontsize=14)
plt.ylabel('Power', fontsize=14)
plt.grid(alpha=0.5)
plt.legend(fontsize=12)
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()