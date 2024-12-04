import numpy as np
from sklearn.datasets import fetch_openml 
from sklearn.ensemble import RandomForestClassifier
from Adadetect import AdaDetectERM 

dataset = fetch_openml(name='creditcard', version=1, as_frame=False)
X = dataset.data
y = dataset.target.astype(np.float)

#test sample 
outlr, inlr = X[y==1], X[y==0]
m1 = 100
m0 = 900 
test1, test0 = outlr[:m1], inlr[:m0]
x = np.concatenate([test0, test1]) 

#NTS
n=5000
xnull = inlr[m0:m0+n]
print(xnull)

#apply AdaDetect with ERM approach, with k=4000 (see notations of the paper)
level=0.05
proc = AdaDetectERM(scoring_fn = RandomForestClassifier(max_depth=10),
                                 split_size=4000/5000) 
rej = proc.apply(x, level, xnull) #gives the rejection set 
print(rej)