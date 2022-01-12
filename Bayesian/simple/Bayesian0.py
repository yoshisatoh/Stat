#################### Bayesian Statistics in Python ####################
#
#  (C) 2022, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2022/01/12
# Last Updated: 2022/01/12
#
#
# Github:
# https://github.com/yoshisatoh/Stat/blob/main/Bayesian/simple/Bayesian0.py
'''
References

1. Bayesian Statistics in Python
https://statsthinking21.github.io/statsthinking21-python/10-BayesianStatistics.html

'''




########## install Python libraries
#
# pip on your Terminal on MacOS (or Command Prompt on Windows) might not work.
#pip install nhanes
#pip install rpy2
#
# If that's the case, then try:
#pip install --upgrade nhanes --trusted-host pypi.org --trusted-host files.pythonhosted.org
#pip install --upgrade rpy2 --trusted-host pypi.org --trusted-host files.pythonhosted.org
#
# If it's successful, then you can repeat the same command for other libraries (i.e., numpy, pandas, scipy, and matplotlib).
#
#
# In some Python 3 environment, the following might work
##### pip3 install --upgrade nhanes
##### pip3 install --upgrade rpy2




########## [1] Applying Bayes’ theorem: A simple example

'''
Sensitivity (True Positive Rate) refers to the proportion of those who have the condition (when judged by the ‘Gold Standard’) that received a positive result on this test.
Specificity (True Negative Rate) refers to the proportion of those who do not have the condition (when judged by the ‘Gold Standard’) that received a negative result on this test.

On the contrary,
False Positive: It is evaluated as Positive (when juddged by the ‘Gold Standard’), but it actually iS FALSE.
False Negative: It is evaluated as Negative (when juddged by the ‘Gold Standard’), but it actually iS POSITIVE.
'''


##### Arbitrary Parameters (You can choose values which suit your circumstances.)
#
#
#True Positive Rate (Likelihood: This is an estimate, but NOT necessary the truth, the whole truth, and nothing but the truth as actual True Positive Rate is unknown.)
sensitivity = 0.90
likelihood = sensitivity  # p(post-alert investigation | alert triggered)
#
#
#True Negative Rate
specificity = 0.99
#
#
'''
Let’s say observed rate of trade surverillance alerts which actually are valid and significant alerts is 7.4%.
That is, 7.4% of all the triggered alerts are true positives and worth investigating.
Please note that fired, but not significant alerts are regarded as negative in this case.
Thus, our prior probability that alerts actually has significanse is 0.074. 
'''
prior = 0.074
#
#####


#marginal_likelihood
#= sensitivity * prior + (1 - specificity) * (1 - prior)
#= (True Positive) * (Positive) + (1- True Negative) * (Negative)
#= (True Positive) * (Positive) + ((False Negative) * (Negative))
#= All Positive Cases including False Negatives
#
marginal_likelihood = sensitivity * prior + (1 - specificity) * (1 - prior)


#(True Positive) * prior = posterior * marginal_likelihood
#likelihood      * prior = posterior * marginal_likelihood
posterior = (likelihood * prior) / marginal_likelihood
#
#prior = True Positives + False Positives
#
#posterior
print(posterior)
#0.8779330345373055




import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt


def compute_posterior(prior, sensitivity, specificity):
    likelihood = sensitivity  # p(post-alert investigation | alert triggered) 
    marginal_likelihood = sensitivity * prior + (1 - specificity) * (1 - prior)
    posterior = (likelihood * prior) / marginal_likelihood
    return(posterior)


prior_values = np.arange(0.001, 0.5, 0.001)
posterior_values = compute_posterior(prior_values, sensitivity, specificity)

plt.plot(prior_values, posterior_values)
plt.xlabel('prior')
_ = plt.ylabel('posterior')
#
plt.savefig('Fig1.png')
plt.show()
'''
This figure highlights a very important general point.
Even when the trade surveillance and alerting have high specificity (True Negative Rate),
if a condition (=alert generation) is rare then most positive results will be false positives.
'''




########## [2] Estimating posterior distributions
'''
In this example we will look at how to estimate entire posterior distributions, rather than single points of estimates.
We apply alert rules to (=test) 100 orders/trades, and found that 64 of them trigerred alerts (positives).
What we want to estimate is the probability distribution for the proportion of alerts, given the data.
For simplicity, we started with a uniform prior; that is, all proprtions of alerts are equally likely to begin with.
In addition, we will use a discrete probability distribution; that is, we will estimate the posterior probabiilty for each particular proportion of alerts, in steps of 0.01.
This greatly simplifies the math and still retains the main idea.
'''


#+
##### Arbitrary Parameters (You can choose values which suit your circumstances.)
#
num_alerts = 64
num_tested = 100
#
#####

bayes_df = pd.DataFrame({'proportion': np.arange(0.0, 1.01, 0.01)})

# compute the binomial likelihood of the observed data for each
# possible value of proportion
bayes_df['likelihood'] = scipy.stats.binom.pmf(num_alerts,
                                               num_tested,
                                               bayes_df['proportion'])
# The prior is equal for all possible values
bayes_df['prior'] = 1 / bayes_df.shape[0]

# compute the marginal likelihood by adding up the likelihood of each possible proportion times its prior probability.

marginal_likelihood = (bayes_df['likelihood'] * bayes_df['prior']).sum()

bayes_df['posterior'] = (bayes_df['likelihood'] * bayes_df['prior']) / marginal_likelihood

# plot the likelihood, prior, and posterior

plt.plot(bayes_df['proportion'], bayes_df['likelihood'], label='likelihood')
plt.plot(bayes_df['proportion'], bayes_df['prior'], label='prior')
plt.plot(bayes_df['proportion'], bayes_df['posterior'],
         'k--', label='posterior')

plt.legend()
#
plt.savefig('Fig2a.png')
plt.show()

#-

'''
The plot shows that the posterior and likelihood are virtually identical, which is due to the fact that the prior is uniform across all possible values. Now let’s look at a case where the prior is not uniform. Let’s say that we now run a larger study of 1000 people with the same treatment, and we find that 312 of the 1000 individuals respond to the treatment. In this case, we can use the posterior from the earlier study of 100 people as the prior for our new study. This is what we sometimes refer to as Bayesian updating.
'''




#+

##### Arbitrary Parameters (You can choose values which suit your circumstances.)
#
num_alerts = 32
num_tested = 100
#
#num_alerts = 64
#num_tested = 100
#
#num_alerts = 312
#num_tested = 1000
#
#num_alerts = 3120
#num_tested = 10000
#
#####


# copy the posterior from the previous analysis and rename it as the prior

study2_df = bayes_df[['proportion', 'posterior']].rename(columns={'posterior': 'prior'})

# compute the binomial likelihood of the observed data for each
# possible value of proportion

study2_df['likelihood'] = scipy.stats.binom.pmf(num_alerts,
                                               num_tested,
                                               study2_df['proportion'])

# compute the marginal likelihood by adding up the likelihood of each possible proportion times its prior probability.

marginal_likelihood = (study2_df['likelihood'] * study2_df['prior']).sum()

study2_df['posterior'] = (study2_df['likelihood'] * study2_df['prior']) / marginal_likelihood

# plot the likelihood, prior, and posterior

plt.plot(study2_df['proportion'], study2_df['likelihood'], label='likelihood')
plt.plot(study2_df['proportion'], study2_df['prior'], label='prior')
plt.plot(study2_df['proportion'], study2_df['posterior'],
         'k--', label='posterior')

plt.legend()
#
plt.savefig('Fig2b.png')
plt.show()


#-

'''
Here we see two important things. First, we see that the prior is substantially wider than the likelihood, which occurs because there is much more data going into the likelihood (1000 data points) compared to the prior (100 data points), and more data reduces our uncertainty. Second, we see that the posterior is much closer to the value observed for the second study than for the first, which occurs for the same reason — we put greater weight on the estimate that is more precise due to a larger sample.

'''



'''
########## Bayes factors

#There are no convenient off-the-shelf tools for estimating Bayes factors using Python, so we will use the rpy2 package to access the BayesFactor library in R. Let’s compute a Bayes factor for a T-test comparing the amount of reported alcohol computing between smokers versus non-smokers. First, let’s set up the NHANES data and collect a sample of 150 smokers and 150 nonsmokers.

import os
os.environ['R_HOME'] = '(set R_HOME here)'


#+
from nhanes.load import load_NHANES_data
nhanes_data = load_NHANES_data()
adult_nhanes_data = nhanes_data.query('AgeInYearsAtScreening > 17')
rseed = 1

# clean up smoking variables
adult_nhanes_data.loc[adult_nhanes_data['SmokedAtLeast100CigarettesInLife'] == 0, 'DoYouNowSmokeCigarettes'] = 'Not at all'
adult_nhanes_data.loc[:, 'SmokeNow'] = adult_nhanes_data['DoYouNowSmokeCigarettes'] != 'Not at all'

# Create average alcohol consumption variable between the two dietary recalls
adult_nhanes_data.loc[:, 'AvgAlcohol'] = adult_nhanes_data[['AlcoholGm_DR1TOT', 'AlcoholGm_DR2TOT']].mean(1)
adult_nhanes_data = adult_nhanes_data.dropna(subset=['AvgAlcohol'])

sample_size_per_group = 150

nonsmoker_sample = adult_nhanes_data.query('SmokeNow == False').sample(sample_size_per_group, random_state=rseed)[['SmokeNow', 'AvgAlcohol']]
smoker_sample = adult_nhanes_data.query('SmokeNow == True').sample(sample_size_per_group, random_state=rseed)[['SmokeNow', 'AvgAlcohol']]

full_sample = pd.concat((nonsmoker_sample, smoker_sample))
full_sample.loc[:, 'SmokeNow'] = full_sample['SmokeNow'].astype('int')
full_sample.groupby('SmokeNow').mean()
#-




#+

# import the necessary functions from rpy2
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()

# import the BayesFactor package
BayesFactor = importr('BayesFactor')

# import the data frames into the R workspace
robjects.globalenv["smoker_sample"] = smoker_sample
robjects.globalenv["nonsmoker_sample"] = nonsmoker_sample

# perform the standard t-test
ttest_output = r('print(t.test(smoker_sample$AvgAlcohol, nonsmoker_sample$AvgAlcohol, alternative="greater"))')

# compute the Bayes factor
r('bf = ttestBF(y=nonsmoker_sample$AvgAlcohol, x=smoker_sample$AvgAlcohol, nullInterval = c(0, Inf))')
r('print(bf[1]/bf[2])')

#-

'''