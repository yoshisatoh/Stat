#################### Estimating Probabilities with Bayesian Inference in Python ####################
#
#  (C) 2022, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2022/01/10
# Last Updated: 2022/01/10
#
#
# Github:
# https://github.com/yoshisatoh/Stats/tree/main/Bayesian/Bayesian1.py
'''
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

Let's say there is a series of 6 OBSERVATIONS: 3 lions, 2 tigers, and 1 bear

From this observation data, we want to estimate the prevalence of each species at a wildlife preserve.

That is, we're going to evaluate the POSTERIOR probability of seeing each species given the observation data above.

Before we begin, we establish our assumptions here:
1. Each observation of one species is an independent trial.
2. Our initial (PRIOR) belief is each species is equally represented.

Your PRIOR will be updated to POSTERIOR based on your OBSERVATIONS.

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
This could be, say, recent observations.

Prior (e.g., Your prior belief, observations in the past):
α = alpha = [lion, tiger, bear] = [1, 1, 1]
Namely,
p(prior, lion)  = 1/3
p(prior, tiger) = 1/3 
p(prior, bear)  = 1/3

----------
Multinomial Distribution

It describes a situation in which we have n independent trials, each with k possible outcomes. 

In this case,
n = 6 (independent trials/observation, i.e., total number of recent observations)
k = 3 (three types of observations, i.e., lion, tiger, and bear)

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




########## install Anaconda
#https://www.anaconda.com/products/individual




########## install Python libraries
#
# pip on your Terminal on MacOS (or Command Prompt on Windows) might not work.
#pip install pymc3
#conda install pymc3
#conda install arviz
#conda install matplotlib
#
# If that's the case, then try:
#pip install --upgrade pymc3 --trusted-host pypi.org --trusted-host files.pythonhosted.org
#
# If it's successful, then you can repeat the same command for other libraries (i.e., numpy, pandas, and matplotlib.pyplot).
#
#
# In some environment, the following might work
##### pip3 install --upgrade pymc3
##### pip3 install git+https://github.com/pymc-devs/pymc3
##### pip3 install git+https://github.com/theano/theano




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
# Download utils.py from the following GitHub page and then save it on the same directory of this script:
#https://github.com/WillKoehrsen/probabilistic-programming/blob/master/utils.py


'''
import os
# one of
os.environ['MKL_THREADING_LAYER'] = 'sequential'
os.environ['OMP_NUM_THREADS'] = '1'
#os.environ['MKL_THREADING_LAYER'] = 'GNU'
#Source:
#https://github.com/pymc-devs/pymc/issues/3140
'''




########## Problem Specifics

'''
We'll mainly use one version of the hyperparameters, α = alpha = [lion, tiger, bear] = [1, 1, 1].
We'll also try some other values to see how that changes the problem.
Remember, altering the hyperparameters is like changing our confidence in our initial beliefs.
'''

# observations
animals = ['lions', 'tigers', 'bears']
k_num = len(animals)    #3
#
c = np.array([3, 2, 1])
c_num = c.sum()    #6    sum(c) also works


# hyperparameters (initially all equal)
alphas = np.array([1, 1, 1])
#
alphas_num = alphas.sum()    #3    sum(alphas) also works


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

with pm.Model() as model:
    # Parameters of the Multinomial are from a Dirichlet
    #parameters = pm.Dirichlet('parameters', a=alphas, shape=3)
    parameters = pm.Dirichlet('parameters', a=alphas, shape=k_num)
    #
    # Observed data is from a Multinomial distribution
    #observed_data = pm.Multinomial('observed_data', n=6, p=parameters, shape=3, observed=c)
    observed_data = pm.Multinomial('observed_data', n=c_num, p=parameters, shape=k_num, observed=c)
'''
if __name__ == '__main__':
    with pm.Model() as model:
        # Parameters of the Multinomial are from a Dirichlet
        parameters = pm.Dirichlet('parameters', a=alphas, shape=3)
        # Observed data is from a Multinomial distribution
        observed_data = pm.Multinomial('observed_data', n=6, p=parameters, shape=3, observed=c)
#Source: https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
'''

#model
print(model)




##### Sampling from the Model
'''
The cell below samples 1000 draws from the posterior in 2 chains. We use 500 samples for tuning which are discarded.
This means that for each random variable in the model - the parameters - we will have 2000 values drawn from the posterior distribution.
'''


with model:
    # Sample from the posterior
    #trace = pm.sample(draws=1000, chains=2, tune=500, discard_tuned_samples=True)
    #trace = pm.sample(draws=1000, chains=2, cores=4, tune=500, discard_tuned_samples=True)
    trace = pm.sample(draws=1000, chains=2, tune=500, discard_tuned_samples=True, cores=1)    # set cores=1 to avoid errors
'''
if __name__ == '__main__':
    with model:
        # Sample from the posterior
        #trace = pm.sample(draws=1000, chains=2, tune=500, discard_tuned_samples=True)
        #trace = pm.sample(draws=1000, tune=500, cores=1)
        trace = pm.sample(draws=1000, tune=500, cores=4)
        #trace = pm.sample(draws=1000, tune=500, chains=1, cores=2)
#Sources:
#https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
#https://discourse.pymc.io/t/error-during-run-sampling-method/2522/4
'''




##### Inspecting Results
summary = pm.summary(trace)
summary.index = animals
#summary
print(summary)

'''
Sampling 2 chains for 500 tune and 1_000 draw iterations (1_000 + 2_000 draws total) took 21 seconds.0<00:00 Sampling chain 1, 0 divergences]
         mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
lions   0.445  0.159   0.155    0.726      0.004    0.003    1418.0     856.0    1.0
tigers  0.337  0.152   0.084    0.617      0.004    0.003    1360.0     990.0    1.0
bears   0.218  0.130   0.018    0.455      0.004    0.003    1344.0    1158.0    1.0
'''


# Samples
trace_df = pd.DataFrame(trace['parameters'], columns = animals)
#trace_df.head()
print(trace_df.head())
'''
      lions    tigers     bears
0  0.448454  0.441321  0.110225
1  0.472099  0.314835  0.213066
2  0.486489  0.372367  0.141144
3  0.646083  0.081343  0.272574
4  0.489364  0.109576  0.401060
'''


#trace_df.shape
print(trace_df.shape)
#(2000, 3)


#For a single point estimate, we can use the mean of the samples.
#
# For probabilities use samples after burn in
pvals = trace_df.iloc[:, :3].mean(axis = 0)
display_probs(dict(zip(animals, pvals)))
'''
Species: lions    Prevalence: 44.47%.
Species: tigers   Prevalence: 33.75%.
Species: bears    Prevalence: 21.78%.
'''


#summary.iloc[:, 3:5]
print(summary.iloc[:, 3:5])




########## Diagnostic Plots

##### Posterior Plot

'''
ax = pm.plot_posterior(trace, varnames = ['parameters'], 
                       figsize = (20, 10), edgecolor = 'k');
'''
'''
ax = pm.plots.plot_posterior(trace, varnames = ['parameters'], 
                       figsize = (20, 10), edgecolor = 'k');
'''
#ax = pm.plots.plot_posterior(trace, None, figsize = (20, 10));
ax = pm.plot_posterior(trace, None, figsize = (20, 10));
                       
#plt.rcParams['font.size'] = 22
plt.rcParams['font.size'] = 18

for i, a in enumerate(animals):
    ax[i].set_title(a);
#
plt.savefig('Fig3.png')
plt.show()
#




##### Traceplot

prop_cycle = plt.rcParams['axes.prop_cycle']
cs = [x['color'] for x in list(prop_cycle)]

#ax = pm.traceplot(trace, varnames = ['parameters'], figsize = (20, 8), combined = True);
ax = pm.traceplot(trace, None, figsize = (20, 8), combined = True);
ax[0][0].set_title('Posterior Probability Distribution'); ax[0][1].set_title('Trace Samples');
ax[0][0].set_xlabel('Probability'); ax[0][0].set_ylabel('Density');
ax[0][1].set_xlabel('Sample number');
add_legend(ax[0][0])
add_legend(ax[0][1])

#
plt.savefig('Fig4.png')
plt.show()
#




##### Maximum A Posteriori Result with PyMC3

with model:
    # Find the maximum a posteriori estimate
    map_ = pm.find_MAP()
    
display_probs(dict(zip(animals, map_['parameters'])))
'''
Species: lions    Prevalence: 50.00%.
Species: tigers   Prevalence: 33.33%.
Species: bears    Prevalence: 16.67%.
'''
#The MAP estimates are exactly the same as the observations. These are also the results that a frequentist would come up with!




##### Sample From Posterior

with model:
    #samples = pm.sample_ppc(trace, samples = 1000)
    samples = pm.sample_posterior_predictive(trace, samples = 1000)
    
dict(zip(animals, samples['observed_data'].mean(axis = 0)))

sample_df = pd.DataFrame(samples['observed_data'], columns = animals)

plt.figure(figsize = (22, 10))
for i, animal in enumerate(sample_df):
    plt.subplot(1, 3, i+1)
    sample_df[animal].value_counts().sort_index().plot.bar(color = 'r');
    plt.xticks(range(7), range(7), rotation = 0);
    plt.xlabel('Number of Times Seen'); plt.ylabel('Occurences');
    plt.title(f'1000 Samples for {animal}');
#
plt.savefig('Fig5.png')
plt.show()
#




##### Dirichlet Distribution

#draw_pdf_contours(Dirichlet(6 * alphas))
draw_pdf_contours(Dirichlet(c_num * alphas))
#
annotate_plot()
#
plt.savefig('Fig6a.png')
plt.show()
#


#draw_pdf_contours(Dirichlet(6 * pvals))
draw_pdf_contours(Dirichlet(c_num * pvals))
#
annotate_plot();
#
plt.savefig('Fig6b.png')
plt.show()
#


draw_pdf_contours(Dirichlet([0.1, 0.1, 0.1]))
annotate_plot();
#
plt.savefig('Fig6c.png')
plt.show()
#




##### Next Observation

# Draw from the multinomial
next_obs = np.random.multinomial(n = 1, pvals = pvals, size = 10000)

# Data manipulation
next_obs = pd.melt(pd.DataFrame(next_obs, columns = ['Lions', 'Tigers', 'Bears'])).\
            groupby('variable')['value'].\
            value_counts(normalize=True).to_frame().\
             rename(columns = {'value': 'total'}).reset_index()
next_obs = next_obs.loc[next_obs['value'] == 1]

# Bar plot
#next_obs.set_index('variable')['total'].plot.bar(figsize = (8, 6));
next_obs.set_index('variable')['total'].plot.bar(figsize = (10, 12));
plt.title('Next Observation Likelihood');
plt.ylabel('Likelihood'); plt.xlabel('');
#
plt.savefig('Fig7.png')
plt.show()
#


next_obs.iloc[:, [0, 2]]




##### More Observations

c = np.array([[3, 2, 1],
              [2, 3, 1],
              [3, 2, 1],
              [2, 3, 1]])

with pm.Model() as model:
    # Parameters are a dirichlet distribution
    #parameters = pm.Dirichlet('parameters', a=alphas, shape=3)
    parameters = pm.Dirichlet('parameters', a=alphas, shape=k_num)
    #
    # Observed data is a multinomial distribution
    '''
    observed_data = pm.Multinomial(
        'observed_data', n=6, p=parameters, shape=3, observed=c)    
    '''
    observed_data = pm.Multinomial('observed_data', n=c_num, p=parameters, shape=k_num, observed=c)    
    #
    #trace = pm.sample(draws=1000, chains=2, tune=500, discard_tuned_samples=True)
    trace = pm.sample(draws=1000, chains=2, tune=500, discard_tuned_samples=True, cores=1)



summary = pm.summary(trace)
summary.index = animals
#summary
print(summary)
'''
         mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
lions   0.405  0.088   0.244    0.566      0.002    0.002    1541.0    1200.0    1.0
tigers  0.409  0.092   0.244    0.588      0.002    0.002    1533.0    1103.0    1.0
bears   0.185  0.073   0.060    0.321      0.002    0.001    1462.0    1233.0    1.0
'''


'''
ax = pm.plot_posterior(trace, varnames = ['parameters'], 
                       figsize = (20, 10), edgecolor = 'k');
'''
ax = pm.plot_posterior(trace, None, figsize = (20, 10));

plt.rcParams['font.size'] = 22
for i, a in enumerate(animals):
    ax[i].set_title(a);
    
plt.suptitle('Posterior with More Observations', y = 1.05);
#
plt.savefig('Fig8.png')
plt.show()
#




########## Increasing/Decreasing Confidence in Hyperparameters

# observations
animals = ['lions', 'tigers', 'bears']
c = np.array([3, 2, 1])

def sample_with_priors(alphas):
    """Sample with specified hyperparameters"""
    with pm.Model() as model:
        # Probabilities for each species
        parameters = pm.Dirichlet('parameters', a=alphas, shape=3)
        # Observed data is a multinomial distribution with 6 trials
        observed_data = pm.Multinomial(
            'observed_data', n=6, p=parameters, shape=3, observed=c)
        #
        #trace = pm.sample(draws=1000, chains=2, tune=500, discard_tuned_samples=True)
        trace = pm.sample(draws=1000, chains=2, tune=500, discard_tuned_samples=True, cores=1)
        #
    return trace

trace_dict = {}
for alpha_array in [np.array([0.1, 0.1, 0.1]), np.array([1, 1, 1]),
                    np.array([5, 5, 5]), np.array([15, 15, 15])]:
    trace_dict[str(alpha_array[0])] = sample_with_priors(alpha_array)




plt.figure(figsize = (20, 24))

for ii, (alpha, trace) in enumerate(trace_dict.items()):
    plt.subplot(4, 1, ii + 1)
    array = trace['parameters']
    for jj, animal in enumerate(animals):
        sns.kdeplot(array[:, jj], label = f'{animal}')
    plt.legend();
    plt.xlabel('Probability'); plt.ylabel('Density')
    plt.title(f'Alpha = {alpha}');
    plt.xlim((0, 1));
    
plt.tight_layout();

plt.savefig('Fig9.png')
plt.show();




########## Comparison of Beliefs

#α = alpha = [lion, tiger, bear] = [0.1, 0.1, 0.1]
prior = '0.1'
trace = trace_dict[prior]

'''
ax = pm.plot_posterior(trace, varnames = ['parameters'], 
                       figsize = (20, 10), edgecolor = 'k');
'''
ax = pm.plot_posterior(trace, None, 
                       figsize = (20, 10));

plt.rcParams['font.size'] = 22
for i, a in enumerate(animals):
    ax[i].set_title(a);
    
plt.suptitle(f'{prior} Prior', y = 1.05);

plt.savefig('Fig10a.png')
plt.show();


summary = pm.summary(trace)
summary.index = animals
#summary
print(summary)
'''
         mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
lions   0.499  0.182   0.179    0.830      0.005    0.004    1347.0    1095.0    1.0
tigers  0.325  0.171   0.051    0.639      0.005    0.003    1340.0    1054.0    1.0
bears   0.176  0.139   0.000    0.437      0.004    0.003    1144.0     873.0    1.0
'''




#α = alpha = [lion, tiger, bear] = [15, 15, 15]
prior = '15'
trace = trace_dict[prior]

'''
ax = pm.plot_posterior(trace, varnames = ['parameters'], 
                       figsize = (20, 10), edgecolor = 'k');
'''
ax = pm.plot_posterior(trace, None, 
                       figsize = (20, 10));

plt.rcParams['font.size'] = 22
for i, a in enumerate(animals):
    ax[i].set_title(a);
    
plt.suptitle(f'{prior} Prior', y = 1.05);

plt.savefig('Fig10b.png')
plt.show();


summary = pm.summary(trace)
summary.index = animals
#summary
print(summary)
'''
         mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
lions   0.352  0.070   0.218    0.481      0.002    0.002     811.0    1006.0    1.0
tigers  0.334  0.066   0.215    0.465      0.002    0.002     948.0    1028.0    1.0
bears   0.314  0.064   0.194    0.428      0.002    0.001    1553.0    1123.0    1.0
'''




########## Conclusions

#α = alpha = [lion, tiger, bear] = [1, 1, 1]
prior = '1'
trace = trace_dict[prior]

'''
ax = pm.plot_posterior(trace, varnames = ['parameters'], 
                       figsize = (20, 10), edgecolor = 'k');
'''
ax = pm.plot_posterior(trace, None, 
                       figsize = (20, 10));

plt.rcParams['font.size'] = 22
for i, a in enumerate(animals):
    ax[i].set_title(a);
    
plt.suptitle('Posterior Plot', y = 1.05);

plt.savefig('Fig10c.png')
plt.show();


summary = pm.summary(trace)
summary.index = animals
#summary
print(summary)
'''
         mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
lions   0.436  0.156   0.150    0.706      0.004    0.003    1606.0    1162.0    1.0
tigers  0.341  0.151   0.094    0.624      0.004    0.003    1233.0    1142.0    1.0
bears   0.223  0.132   0.019    0.475      0.004    0.003    1329.0    1182.0    1.0
'''