'''
Estimating Probabilities with Bayesian Inference


References

Estimating Probabilities with Bayesian Modeling in Python
https://towardsdatascience.com/estimating-probabilities-with-bayesian-modeling-in-python-7144be007815

https://github.com/WillKoehrsen/probabilistic-programming/blob/master/Estimating%20Probabilities%20with%20Bayesian%20Inference.ipynb

'''


'''
A series of observations: 3 lions, 2 tigers, and 1 bear,

and from this data, we want to estimate the prevalence of each species at the wildlife preserve.
(posterior probability of seeing each species given the observation data)

Before we begin we want to establish our assumptions:
1. Treat each observation of one species as an independent trial.
2. Our initial (prior) belief is each species is equally represented.


-----

Model

The overall system is as follows (each part will be explained):
1. The underlying model is a multinomial distribution with parameters pk
2. The prior distribution of pk is a Dirichlet Distribution
3. The α vector is a parameter of the prior Dirichlet Distribution, hence a hyperparameter

α   = (α1, ..., αk) = concentration hyperparameter
p|α = (p1, ..., pk) = Dir(K, α)    p1 + ... + pk = 1
X|p = (x1, ..., xk) = Mult(K, p)

Our goal is to find 

p (posterior, lion)
p (posterior, tiger)
p (posterior, bear)


given the
observation vector c = [c(lion), c(tiger), c(bear)] = [3, 2, 1] 


α = alpha = [lion, tiger, bear] = [1, 1, 1] 

p(prior, lion)  = 1/3
p(prior, tiger) = 1/3 
p(prior, bear)  = 1/3

-----
Multinomial Distribution
which describes a situation in which we have n independent trials, each with k possible outcomes. 

In this case,
n = 6 (independent trials)
k = 3 (lion, tiger, bear)

-----
Dirichlet Distribution

The prior for a multinomial distribution in Bayesian statistics is a Dirichlet distribution. 

The Dirichlet distribution is characterized by α, the concentration hyperparameter vector.

The α vector is a hyperparameter, a parameter of a prior distribution. This vector in turn could have its own prior distribution which is called a hyperprior. We won't use a hyperprior, but instead will only specify the hyperparameters.
'''




########## Hyperparameters and Prior Beliefs

##### pip3 install --upgrade mkl   
##### pip3 install git+https://github.com/theano/theano
##### pip3 install git+https://github.com/pymc-devs/pymc3


import pandas as pd
import numpy as np

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 22
#%matplotlib inline

from matplotlib import MatplotlibDeprecationWarning

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)

import pymc3 as pm

# Helper functions
from utils import draw_pdf_contours, Dirichlet, plot_points, annotate_plot, add_legend, display_probs