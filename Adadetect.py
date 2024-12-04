import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import GridSearchCV, ParameterGrid
from functools import reduce

# ------------------ EmpBH --------------------------

def EmpBH(null_statistics, test_statistics, level):
    """
    Algorithm 1 of "Semi-supervised multiple testing", Roquain & Mary : faster than computing p-values and applying BH

    test_statistics: scoring function evaluated at the test sample i.e. g(X_1), ...., g(X_m)
    null_statistics: scoring function evaluated at the null sample that is used for calibration of the p-values i.e. g(Z_k),...g(Z_n)
    level: nominal level 

    Return: rejection set 
    """
    n, m = len(null_statistics), len(test_statistics)

    mixed_statistics = np.concatenate([null_statistics, test_statistics])
    sample_ind = np.concatenate([np.ones(len(null_statistics)), np.zeros(len(test_statistics))])

    sample_ind_sort = sample_ind[np.argsort(-mixed_statistics)] 
    
    #np.argsort(-mixed_statistics) gives the order of the stats in descending order 
    #sample_ind_sort sorts the 1-labels according to this order 

    fdp = 1 
    V = n
    K = m 
    l=m+n

    while (fdp > level and K >= 1):
        l-=1
        if sample_ind_sort[l] == 1:
            V-=1
        else:
            K-=1
        fdp = (V+1)*m / ((n+1)*K) if K else 1 

    test_statistics_sort_ind = np.argsort(-test_statistics)
    return test_statistics_sort_ind[:K]

#----------------------- AdaDetect ----------------------------
class AdaDetectBase(object):
    """
    Base template for AdaDetect procedures to inherit from. 
    """

    def __init__(self, correction_type=None, storey_threshold=0.5):
        """
        correction_type: if 'storey'/'quantile', uses the adaptive AdaDetect procedure with storey/quantile correction
        """
        self.null_statistics = None
        self.test_statistics = None 
        self.correction_type = correction_type
        self.storey_threshold = storey_threshold

    def fit(self, x, level, xnull):
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: none. Sets the values for <null_statistics> / <test_statistics>. 
        """
        #This part depends specifically on the type of AdaDetect procedure: 
        #whether the scoring function g is learned via density estimation, or an ERM approach (PU classification)
        #Thus, it is coded in separate AdaDetectBase objects, see below. 

        pass
    
    def apply(self, x, level, xnull): 
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: rejection set of AdaDetect with scoring function g learned from <x> and <xnull> as per .fit(). 
        """ 
        self.fit(x, level, xnull)
        return EmpBH(self.null_statistics, self.test_statistics, level = level)


class AdaDetectDE(AdaDetectBase):
    """
    AdaDetect procedure where the scoring function is learned by a density estimation approach. There are two possibilities: 
        - Either the null distribution is assumed known, in which case the scoring function is learned on the mixed sample = test sample + NTS. 
        - Otherwise, the NTS is split, and the scoring function is learned separatly on a part of the NTS (to learn the null distribution) and on the remaining mixed sample. 

    Note: one-class classification (approach Bates et. al) can be obtained from this routine: it suffices to define scoring_fn (see below) such that only the first part of the NTS is used. 
    """

    def __init__(self, scoring_fn, f0_known=True, split_size=0.5, correction_type=None, storey_threshold = 0.5):
        AdaDetectBase.__init__(self, correction_type, storey_threshold)
        """
        scoring_fn: A class (estimator) that must have a .fit() and a .score_samples() method, e.g. sklearn's KernelDensity() 
                            The .fit() method takes as input a (training) data sample and may set/modify some parameters of scoring_fn
                            The .score_samples() method takes as input a (test) data sample and should return the log-density for each element, as in sklearn's KernelDensity() 
        The same method is used for learning the null distribution as for the 'mixture distribution' of the test sample mixed with the second part of the NTS ('f_gamma' in the paper). 

        f0_known: boolean, indicates whether the null distribution is assumed known (=True, in that case scoring_fn should use this knowledge, 
        e.g. by returning in its score_samples() method the ratio of a fitted mixture density estimator over the true null density) or not (=False)

        split_size: proportion of the part of the NTS used for fitting g i.e. k/n with the notations of the paper
        """
        self.scoring_fn = scoring_fn
        self.f0_known = f0_known
        self.split_size = split_size
        
    
    def fit(self, x, level, xnull):
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: none. Sets the values for <null_statistics> / <test_statistics> properties (which are properties of any AdaDetectBase object) 
        """
        m = len(x)
        n = len(xnull)

        # learn the scoring function
        if self.f0_known: 
            x_train = np.concatenate([xnull, x])
            self.scoring_fn.fit(x_train)

        else:

            #split the null
            n_null_train = int(self.split_size * n)
            xnull_train = xnull[:n_null_train] #this is set aside for learning the score
            xnull_calib = xnull[n_null_train:] #must NOT be set aside!!! must be mixed in with x to keep control 

            xtrain = np.concatenate([xnull_calib, x])

            self.scoring_fn.fit(x_train = xtrain, x_null_train = xnull_train)

            xnull = xnull_calib

        # compute scores 
        self.test_statistics = self.scoring_fn.score_samples(x)
        self.null_statistics = self.scoring_fn.score_samples(xnull) 


class AdaDetectERM(AdaDetectBase):
    """
    AdaDetect procedure where the scoring function is learned by an ERM approach. 
    """


    def __init__(self, scoring_fn, split_size=0.5, correction_type=None, storey_threshold=0.5):
        AdaDetectBase.__init__(self, correction_type, storey_threshold)
        """
        scoring_fn: A class (estimator) that must have a .fit() and a .predict_proba() or .decision_function() method, e.g. sklearn's LogisticRegression() 
                            The .fit() method takes as input a (training) data sample of observations AND labels <x_train, y_train> and may set/modify some parameters of scoring_fn
                            The .predict_proba() method takes as input a (test) data sample and should return the a posteriori class probabilities (estimates) for each element
        
        split_size: proportion of the part of the NTS used for fitting g i.e. k/n with the notations of the paper
        """

        self.scoring_fn = scoring_fn
        self.split_size = split_size

    def fit(self, x, level, xnull):
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: none. Sets the values for <null_statistics> / <test_statistics> properties (which are properties of any AdaDetectBase object) 
        """
        m = len(x)
        n = len(xnull)

        n_null_train = int(self.split_size * n) 
        xnull_train = xnull[:n_null_train]
        xnull_calib = xnull[n_null_train:]

        x_mix_train = np.concatenate([x, xnull_calib])

        #fit a classifier using xnull_train and x_mix_train
        x_train = np.concatenate([xnull_train, x_mix_train])
        y_train = np.concatenate([np.zeros(len(xnull_train)), np.ones(len(x_mix_train))])
        
        self.scoring_fn.fit(x_train, y_train)

        # compute scores 
        methods_list = ["predict_proba", "decision_function"]
        prediction_method = [getattr(self.scoring_fn, method, None) for method in methods_list]
        prediction_method = reduce(lambda x, y: x or y, prediction_method)

        self.null_statistics = prediction_method(xnull_calib)
        self.test_statistics = prediction_method(x)

        if self.null_statistics.ndim != 1:
            self.null_statistics = self.null_statistics[:,1]
            self.test_statistics = self.test_statistics[:,1]


# Took out the cross-validated version, as we have only one algorithm for simulation


