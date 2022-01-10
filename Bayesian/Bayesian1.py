'''
Estimating Probabilities with Bayesian Inference


References

1.Estimating Probabilities with Bayesian Modeling in Python
https://towardsdatascience.com/estimating-probabilities-with-bayesian-modeling-in-python-7144be007815

2. GitHub
https://github.com/WillKoehrsen/probabilistic-programming/blob/master/Estimating%20Probabilities%20with%20Bayesian%20Inference.ipynb
https://github.com/WillKoehrsen/probabilistic-programming/blob/master/utils.py

3.Bayesian Inference for Dirichlet-Multinomials by Mark Johnson
http://users.cecs.anu.edu.au/~ssanner/MLSS2010/Johnson1.pdf
'''


'''
Background

Let's say there is a series of observations: 3 lions, 2 tigers, and 1 bear

From this observation data, we want to estimate the prevalence of each species at a wildlife preserve.

That is, we're going to evaluate the POSTERIOR probability of seeing each species given the observation data above.

Before we begin, we establish our assumptions here:
1. Each observation of one species is an independent trial.
2. Our initial (PRIOR) belief is each species is equally represented.

----------
Model

The overall system is as follows (each part will be explained):
1. The underlying model is a multinomial distribution with parameters pk
2. The prior distribution of pk is a Dirichlet Distribution
3. The α vector is a parameter of the prior Dirichlet Distribution, hence a hyperparameter

α   = (α1, ..., αk) = concentration hyperparameter
p|α = (p1, ..., pk) = Dir(K, α)    where    p1 + ... + pk = 1
X|p = (x1, ..., xk) = Mult(K, p)

Our goal is to find:
p (posterior, lion)
p (posterior, tiger)
p (posterior, bear)

Given:
observation vector c = [c(lion), c(tiger), c(bear)] = [3, 2, 1] 

Prior:
α = alpha = [lion, tiger, bear] = [1, 1, 1] 
Namely,
p(prior, lion)  = 1/3
p(prior, tiger) = 1/3 
p(prior, bear)  = 1/3

----------
Multinomial Distribution

It describes a situation in which we have n independent trials, each with k possible outcomes. 

In this case,
n = 6 (independent trials/observation)
k = 3 (three types of observation, i.e., lion, tiger, and bear)

----------
Dirichlet Distribution

The prior for a multinomial distribution in Bayesian statistics is a Dirichlet distribution. 

The Dirichlet distribution is characterized by α, the concentration hyperparameter vector.

The α vector is a hyperparameter, a parameter of a prior distribution.
This vector in turn could have its own prior distribution which is called a hyperprior. We won't use a hyperprior, but instead will only specify the hyperparameters.

 It is a multivariate generalization of the beta distribution, hence its alternative name of multivariate beta distribution (MBD).
 Dirichlet distributions are commonly used as prior distributions in Bayesian statistics, and in fact the Dirichlet distribution is the conjugate prior of the categorical distribution and multinomial distribution.
 
 See:
 https://en.wikipedia.org/wiki/Dirichlet_distribution
 https://en.wikipedia.org/wiki/Beta_distribution
'''




########## install Python libraries
#
# pip on your Terminal on MacOS (or Command Prompt on Windows) might not work.
#pip install pymc3
#
# If that's the case, then try:
#pip install --upgrade pymc3 --trusted-host pypi.org --trusted-host files.pythonhosted.org
#
# If it's successful, then you can repeat the same command for other libraries (i.e., numpy, pandas, and matplotlib.pyplot).
#
#
##### pip3 install --upgrade mkl   
##### pip3 install git+https://github.com/theano/theano
##### pip3 install git+https://github.com/pymc-devs/pymc3




########## Hyperparameters and Prior Beliefs

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
'''
Ignore if you see the following warning:

WARNING (theano.configdefaults): g++ not available, if using conda: `conda install m2w64-toolchain`
WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
'''

# Helper functions
from utils import draw_pdf_contours, Dirichlet, plot_points, annotate_plot, add_legend, display_probs
#
# Download utils.py from and save it on the same directory of this script:
#https://github.com/WillKoehrsen/probabilistic-programming/blob/master/utils.py




########## Problem Specifics

'''
We'll mainly use one version of the hyperparameters, α = alpha = [lion, tiger, bear] = [1, 1, 1].
We'll also try some other values to see how that changes the problem.
Remember, altering the hyperparameters is like chaning our confidence in our initial beliefs.
'''

# observations
animals = ['lions', 'tigers', 'bears']
c = np.array([3, 2, 1])

# hyperparameters (initially all equal)
alphas = np.array([1, 1, 1])

# Various hyperparameters
# All are equal, but large numbers, such as 15 for each (45 in total), mean your PRIOR belief in equal distribution is stronger than smaller numbers, say 0.1.
# Alpha is like your observations in the past. If you see large numbers of animals which are equally distributed in in the past, then your PRIOR belief in equal distribution is strong amd not that affected by the recent observation factor c.
alpha_list = [np.array([0.1, 0.1, 0.1]), np.array([1, 1, 1]),
                    np.array([5, 5, 5]), np.array([15, 15, 15])]




########## Expected Value of the POSTERIOR for pk

#The expected value of a Dirichlet-Multinomial Distribution is:
#E[pi|X,α] = (ci+αi)/(N+Σk(αk))

#Using α = alpha = [lion, tiger, bear] = [1, 1, 1] and the observation vector c  = [lion, tiger, bear] = [3, 2, 1], we get the expected prevalances.
#
#p(lion)  = (3+1)/(6+3) = 4/9 = 0.4444.. = 44.4%
#p(tiger) = (2+1)/(6+3) = 3/9 = 0.3333.. = 33.3%
#p(bear)  = (1+1)/(6+3) = 2/9 = 0.2222.. = 22.2%

display_probs(dict(zip(animals, (alphas + c) / (c.sum() + alphas.sum()))))
#display_probs(dict(zip(animals, (4/9, 3/9, 2/9))))

#Now, let's try a few different hyperparameter values (our previous beliefs) and see how that affects the expected value.
values = []
for alpha_new in alpha_list:
    values.append((alpha_new + c) / (c.sum() + alpha_new.sum()))

value_df = pd.DataFrame(values, columns = animals)
value_df['alphas'] = [str(x) for x in alpha_list]
#value_df
print(value_df)


melted = pd.melt(value_df, id_vars = 'alphas', value_name='prevalence',
        var_name = 'species')

plt.figure(figsize = (10, 10))
sns.barplot(x = 'alphas', y = 'prevalence', hue = 'species', data = melted,
            edgecolor = 'k', linewidth = 1.5);
plt.xticks(size = 14); plt.yticks(size = 14)
plt.title('Expected Value');
#
plt.savefig('Fig1.png')
plt.show()
#
#With heavier priors, the data (observations) matters less for POSTERIORS.
#The ultimate choice of the hyperparameters α (alpha) depends on our confidence in our prior belief.




########## Maximum A Posterior Estimation

#The maximum a posterior (MAP) is another point estimate that is equal to the mode of the posterior distribution.
#The MAP for a Dirichlet-Multinomial is:

#argmax(p) p(p|X) = (αi+ci-1)/Σ(αi+ci-1)

#In the case of α = alpha = [lion, tiger, bear] = [1, 1, 1], this becomes the frequency observed in the data as (αi-1) terms in denominator and numerator, resepectively, always get to zero.
#Only observation vector c = [c(lion), c(tiger), c(bear)] = [3, 2, 1] remains in that case.

display_probs(dict(zip(animals, (alphas + c - 1) / sum(alphas + c - 1))))

#display_probs(dict(zip(animals, c / c.sum())))


#Let's look at what happens when we change the hyperparameters.

values = []
for alpha_new in alpha_list:
    values.append((alpha_new + c - 1) / sum(alpha_new + c - 1))

value_df = pd.DataFrame(values, columns = animals)
value_df['alphas'] = [str(x) for x in alpha_list]
#value_df
print(value_df)


melted = pd.melt(value_df, id_vars = 'alphas', value_name='prevalence',
        var_name = 'species')

plt.figure(figsize = (10, 10))
sns.barplot(x = 'alphas', y = 'prevalence', hue = 'species', data = melted,
            edgecolor = 'k', linewidth = 1.5);
plt.xticks(size = 14); plt.yticks(size = 14)
plt.title('Maximum A Posterior Value');
#
plt.savefig('Fig2.png')
plt.show()




########## Bayesian Model

'''
Now we'll get into building and sampling from a Bayesian model.
As a reminder, we are using a multinomial as our model, a Dirichlet distribution as the prior, and a specified hyperparameter vector.
The objective is to find the parameters of the multinomial, pk, which are the probability of each species given the evidence.

(p|X, α)
'''
'''
PyMC3 and MCMC

To solve the problem, we'll build a model in PyMC3 and then use a variant of Markov Chain Monte Carlo (the No-UTurn Sampler specifically) to draw samples from the posterior. With enough samples, the estimate will converge on the true posterior. Along with single point estimates (such as the mean of sampled values), MCMC also gives us built in uncertainty because we get thousands of possible values from the posterior.

Building a model in PyMC3 is simple. Each distribution is specified, along with the required parameters. We assign the observed counts to the observed parameter of the multinomial which in turn has the Dirichlet as the prior.
'''

'''
We'll use
α = alpha = [lion, tiger, bear] = [1, 1, 1]
for our main model.
'''

'''
with pm.Model() as model:
    # Parameters of the Multinomial are from a Dirichlet
    parameters = pm.Dirichlet('parameters', a=alphas, shape=3)
    # Observed data is from a Multinomial distribution
    observed_data = pm.Multinomial(
        'observed_data', n=6, p=parameters, shape=3, observed=c)
'''
if __name__ == '__main__':
    with pm.Model() as model:
        # Parameters of the Multinomial are from a Dirichlet
        parameters = pm.Dirichlet('parameters', a=alphas, shape=3)
        # Observed data is from a Multinomial distribution
        observed_data = pm.Multinomial('observed_data', n=6, p=parameters, shape=3, observed=c)
#Source: https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing

#model
print(model)




##### Sampling from the Model
'''
The cell below samples 1000 draws from the posterior in 2 chains. We use 500 samples for tuning which are discarded. This means that for each random variable in the model - the parameters - we will have 2000 values drawn from the posterior distribution.
'''
'''
with model:
    # Sample from the posterior
    trace = pm.sample(draws=1000, chains=2, tune=500, 
                      discard_tuned_samples=True)
'''


if __name__ == '__main__':
    with model:
        # Sample from the posterior
        trace = pm.sample(draws=1000, chains=2, tune=500, discard_tuned_samples=True)
#Source: https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing



##### Inspecting Results
summary = pm.summary(trace)
summary.index = animals
#summary
print(summary)
