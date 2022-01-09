'''
Bayesian Statistics with Python


To install/upgrade Python libraries, arviz for instance, execute:
python3 -m pip install --upgrade arviz
pip2 install --upgrade arviz


#import numpy as np
#print(np.__version__)
#1.22.0
#
#import scipy
#print(scipy.__version__)
#1.7.3


#You need to install Xcode on MacOS.
g++ --version 



References

Hands On Bayesian Statistics with Python, PyMC3 & ArviZ
https://towardsdatascience.com/hands-on-bayesian-statistics-with-python-pymc3-arviz-499db9a59501

https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv
'''

from scipy import stats
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import pandas as pd
#from theano import shared
from sklearn import preprocessing

print('Running on PyMC3 v{}'.format(pm.__version__))

#data = pd.read_csv('renfe.csv')
data = pd.read_csv('renfe_small.csv')
data.drop('Unnamed: 0', axis = 1, inplace=True)
data = data.sample(frac=0.01, random_state=99)
data.head(3)