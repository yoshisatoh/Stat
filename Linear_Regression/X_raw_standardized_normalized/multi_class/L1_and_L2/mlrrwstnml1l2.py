#################### Multiple (and Single) Linear Regression of Raw Data, Standardized Data, and Normalized Data ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/09/25
# Last Updated: 2021/09/25
#
# Github:
# https://github.com/yoshisatoh/Stat/tree/main/Linear_Regression/X_raw_standardized_normalized/multi_class/L1_and_L2/mlrrwstnml1l2.py
#
#
########## Input Data Files
#
#kc_house_data_updated.csv
#
#
########## Usage Instructions
#
#python3 mlrrwstnml1l2.py (arg_df_name: csv data file) (arg_y_col: dependent/explained variable Y) (arg_test_size: test data size)
#
#For instance,
#
#[Option A] Raw Data X vs Raw Data Y
#python mlrrwstnml1l2.py kc_house_data_updated.csv price 0.25 r
#
#[Option B] Standardized Data X vs Raw Data Y
#python mlrrwstnml1l2.py kc_house_data_updated.csv price 0.25 s
#
#[Option C] Normalized Data X vs Raw Data Y
#python mlrrwstnml1l2.py kc_house_data_updated.csv price 0.25 n
#
#
########## Output Data Files
#
#df_Xconv.describe.txt
#df_train.csv
#df_test.csv
#
#linearModel.txt
#linearModel.df_coef_.csv
#linearModel.ds_intercept_.csv
#linearModel.ds_y_test_pred.csv
#
#ridgeModelChosen.txt
#ridgeModelChosen.df_coef_.csv
#ridgeModelChosen.ds_intercept_.csv
#ridgeModelChosen.ds_y_test_pred.csv
#
#lassoModelChosen.txt
#lassoModelChosen.df_coef_.csv
#lassoModelChosen.ds_intercept_.csv
#lassoModelChosen.ds_y_test_pred.csv
#
#
########## Reference(s)
#
#https://www.geeksforgeeks.org/ml-implementing-l1-and-l2-regularization-using-sklearn/
#https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv
#
#
####################




########## import Python libraries

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean

#from sklearn import datasets
#from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#import statsmodels.api as sm




########## arguments

arg_df_name   = str(sys.argv[1])    # e.g., kc_house_data_updated.csv
arg_y_col     = str(sys.argv[2])    # e.g., price - Y (dependent/explained vairible)
arg_test_size = float(sys.argv[3])    # e.g., 0.25 - test : train = 0.25 : (1-0.25)
arg_X_cov     = str(sys.argv[4])    # e.g., r, s, or n: X data conversion - r(raw), s(standardized), or n(normalized)




########### load csv dataset

# Loading the data into a Pandas DataFrame
df = pd.read_csv(arg_df_name)
 
# Separating the dependent and independent variables
ds_y = df[arg_y_col]    # pandas.Series
df_X = df.drop(arg_y_col, axis = 1)    # pandas.DataFrame




########### set X (independent variables)

if arg_X_cov == 'r':
    #r: raw X data
    print('r: raw X data')
    #
    df_Xconv = df_X    # no conversion
    #
    #
elif arg_X_cov == 's':
    #s: standardized X data
    # Standardized data X (average == 0, standard deviation == 1 for all x)
    print('s: standardized X data')
    #
    scaler = StandardScaler()
    scaler.fit(df_X)
    #
    Xconv    = scaler.transform(df_X)
    df_Xconv = pd.DataFrame(Xconv)
    df_Xconv.columns = df_X.columns
    #
    #    
elif arg_X_cov == 'n':
    #n: normalized X data
    # Normalized data X (all x are normalized to take values from 0 to 1)
    print('n: normalized X data')
    #
    mscaler = MinMaxScaler()
    mscaler.fit(df_X)
    #
    Xconv    = mscaler.transform(df_X)
    df_Xconv = pd.DataFrame(Xconv)
    df_Xconv.columns = df_X.columns
    #
    #    
else:
    print('Speficy X data conversion: r (raw X data), s (standardized X data), or n (normalized X data)')
    sys.exit(1)    #error status = 1
#
#
with open('df_Xconv.describe.txt', 'w') as f:
    print(df_Xconv.describe(), file=f)




########### Dividing the data into train and test datasets
np.random.seed(0)    # fix the data splitting results by specifying n=0
X_train, X_test, y_train, y_test = train_test_split(df_Xconv, ds_y, test_size = arg_test_size)

df_X_train         = pd.DataFrame(X_train)
df_X_train.columns = df_Xconv.columns

df_X_test          = pd.DataFrame(X_test)
df_X_test.columns  = df_Xconv.columns

ds_y_train         = pd.Series(y_train)
ds_y_train.name    = ds_y.name

ds_y_test          = pd.Series(y_test)
ds_y_test.name     = ds_y.name

df_train = pd.merge(ds_y_train, df_X_train, left_index=True, right_index=True)
df_test  = pd.merge(ds_y_test,  df_X_test,  left_index=True, right_index=True)

df_train.to_csv('df_train.csv', sep=',', header=True, index=False)
df_test.to_csv('df_test.csv',   sep=',', header=True, index=False)




########## Multiple Linear Regression (without Regularization)

print('********** Multiple Linear Regression (without Regularization) **********')

# Building and fitting the Linear Regression model
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)
 
# Evaluating the Linear Regression model
print(linearModel.score(X_test, y_test))
#print(linearModel.coef_)
#print(linearModel.intercept_)
#print(linearModel.predict(X_test))

# Output
with open('linearModel.txt', 'w') as f:
    print(linearModel.score(X_test, y_test), file=f)
    #print(linearModel.coef_,                 file=f)
    #print(linearModel.intercept_,            file=f)
#
linearModel.df_coef_         = pd.DataFrame(linearModel.coef_).transpose()
linearModel.df_coef_.columns = df_X.columns
linearModel.df_coef_.to_csv('linearModel.df_coef_.csv', sep=',', header=True, index=False)
#
linearModel.ds_intercept_      = pd.Series(linearModel.intercept_)
linearModel.ds_intercept_.name = 'intercept_'
linearModel.ds_intercept_.to_csv('linearModel.ds_intercept_.csv', sep=',', header=True, index=False)
#
#
#with open('linearModel_X_test_y_pred.txt', 'w') as f:
#    print(linearModel.predict(X_test),       file=f)
#
linearModel.ds_y_test_pred      = pd.Series(linearModel.predict(X_test))
linearModel.ds_y_test_pred.name = ds_y.name
linearModel.ds_y_test_pred.to_csv('linearModel.ds_y_test_pred.csv', sep=',', header=True, index=False)




########## Multiple Linear Regression (with Ridge(L2) Regression)

print('********** Multiple Linear Regression (with Ridge(L2) Regression) **********')

# List to maintain the different cross-validation scores
cross_val_scores_ridge = []
 
# List to maintain the different values of alpha
alpha = []
 
# Loop to compute the different values of cross-validation scores
for i in range(1, 9):
    #ridgeModel = Ridge(alpha = i * 0.25)
    ridgeModel = Ridge(alpha = i)
    #
    ridgeModel.fit(X_train, y_train)
    #
    #scores = cross_val_score(ridgeModel, X, y, cv = 10)
    scores = cross_val_score(ridgeModel, df_X, ds_y, cv = 10)
    #
    avg_cross_val_score = mean(scores)*100
    cross_val_scores_ridge.append(avg_cross_val_score)
    #
    #alpha.append(i * 0.25)
    alpha.append(i)
 
# Loop to print the different values of cross-validation scores
for i in range(0, len(alpha)):
    print(str(alpha[i])+' : '+str(cross_val_scores_ridge[i]))


#From the above output, we can conclude that the best value of alpha for the data is 2.0.
cross_val_scores_ridge_max_value = max(cross_val_scores_ridge)
cross_val_scores_ridge_max_index = cross_val_scores_ridge.index(cross_val_scores_ridge_max_value)
#print(cross_val_scores_ridge_max_index)
#print('Best alpha for Ridge (L2): ' + str(alpha[cross_val_scores_ridge_max_index] * 0.25))    #2.0
#print(alpha)   #3
print('Best alpha for Ridge (L2): ' + str(alpha[cross_val_scores_ridge_max_index]))    #2.0
#exit()


# Building and fitting the Ridge Regression model
#ridgeModelChosen = Ridge(alpha = 2)
#ridgeModelChosen = Ridge(alpha = alpha[cross_val_scores_ridge_max_index] * 0.25)
#print(alpha[cross_val_scores_ridge_max_index])    #3
alpha_chosen = alpha[cross_val_scores_ridge_max_index]
ridgeModelChosen = Ridge(alpha = alpha_chosen)
ridgeModelChosen.fit(X_train, y_train)
 
# Evaluating the Ridge Regression model
print(ridgeModelChosen.score(X_test, y_test))
#print(ridgeModelChosen.coef_)
#print(ridgeModelChosen.intercept_)
#print(ridgeModelChosen.predict(X_test))

# Output
with open('ridgeModelChosen.txt', 'w') as f:
    print(ridgeModelChosen.score(X_test, y_test), file=f)
    #print(ridgeModelChosen.coef_,                 file=f)
    #print(ridgeModelChosen.intercept_,            file=f)
#
ridgeModelChosen.df_coef_         = pd.DataFrame(ridgeModelChosen.coef_).transpose()
ridgeModelChosen.df_coef_.columns = df_X.columns
ridgeModelChosen.df_coef_.to_csv('ridgeModelChosen.df_coef_.csv', sep=',', header=True, index=False)
#
ridgeModelChosen.ds_intercept_      = pd.Series(ridgeModelChosen.intercept_)
ridgeModelChosen.ds_intercept_.name = 'intercept_'
ridgeModelChosen.ds_intercept_.to_csv('ridgeModelChosen.ds_intercept_.csv', sep=',', header=True, index=False)
#
#
#with open('ridgeModelChosen_X_test_y_pred.txt', 'w') as f:
#    print(ridgeModelChosen.predict(X_test),       file=f)
ridgeModelChosen.ds_y_test_pred      = pd.Series(ridgeModelChosen.predict(X_test))
ridgeModelChosen.ds_y_test_pred.name = ds_y.name
ridgeModelChosen.ds_y_test_pred.to_csv('ridgeModelChosen.ds_y_test_pred.csv', sep=',', header=True, index=False)




########## Multiple Linear Regression (with Lasso(L1) Regression)

print('********** Multiple Linear Regression (with Lasso(L1) Regression) **********')

# List to maintain the cross-validation scores
cross_val_scores_lasso = []
 
# List to maintain the different values of Lambda
Lambda = []
 
# Loop to compute the cross-validation scores
for i in range(1, 9):
    #lassoModel = Lasso(alpha = i * 0.25, tol = 0.0925)
    lassoModel = Lasso(alpha = i, tol = 0.0925)
    lassoModel.fit(X_train, y_train)
    #
    #scores = cross_val_score(lassoModel, X, y, cv = 10)
    scores = cross_val_score(lassoModel, df_X, ds_y, cv = 10)
    #
    avg_cross_val_score = mean(scores)*100
    cross_val_scores_lasso.append(avg_cross_val_score)
    #Lambda.append(i * 0.25)
    Lambda.append(i)
 
# Loop to print the different values of cross-validation scores
for i in range(0, len(alpha)):
    print(str(alpha[i])+' : '+str(cross_val_scores_lasso[i]))


#From the above output, we can conclude that the best value of alpha is 2.
cross_val_scores_lasso_max_value = max(cross_val_scores_lasso)
cross_val_scores_lasso_max_index = cross_val_scores_lasso.index(cross_val_scores_lasso_max_value)
#print(cross_val_scores_lasso_max_index)
#print('Best alpha for Lasso (L1): ' + str(alpha[cross_val_scores_lasso_max_index] * 0.25))    #2.0
print('Best alpha for Lasso (L1): ' + str(alpha[cross_val_scores_lasso_max_index]))    #2.0


# Building and fitting the Lasso Regression Model
#lassoModelChosen = Lasso(alpha = 2, tol = 0.0925)
#lassoModelChosen = Lasso(alpha = alpha[cross_val_scores_lasso_max_index] * 0.25, tol = 0.0925)
alpha_chosen = alpha[cross_val_scores_lasso_max_index]
lassoModelChosen = Lasso(alpha = alpha_chosen, tol = 0.0925)
lassoModelChosen.fit(X_train, y_train)


# Evaluating the Lasso Regression model
print(lassoModelChosen.score(X_test, y_test))
#print(lassoModelChosen.coef_)
#print(lassoModelChosen.intercept_)
#print(lassoModelChosen.predict(X_test))

# Output
with open('lassoModelChosen.txt', 'w') as f:
    print(lassoModelChosen.score(X_test, y_test), file=f)
    #print(lassoModelChosen.coef_,                 file=f)
    #print(lassoModelChosen.intercept_,            file=f)
#
lassoModelChosen.df_coef_         = pd.DataFrame(lassoModelChosen.coef_).transpose()
lassoModelChosen.df_coef_.columns = df_X.columns
lassoModelChosen.df_coef_.to_csv('lassoModelChosen.df_coef_.csv', sep=',', header=True, index=False)
#
lassoModelChosen.ds_intercept_      = pd.Series(lassoModelChosen.intercept_)
lassoModelChosen.ds_intercept_.name = 'intercept_'
lassoModelChosen.ds_intercept_.to_csv('lassoModelChosen.ds_intercept_.csv', sep=',', header=True, index=False)
#
#
#with open('lassoModelChosen_X_test_y_pred.txt', 'w') as f:
#    print(lassoModelChosen.predict(X_test),       file=f)
lassoModelChosen.ds_y_test_pred      = pd.Series(lassoModelChosen.predict(X_test))
lassoModelChosen.ds_y_test_pred.name = ds_y.name
lassoModelChosen.ds_y_test_pred.to_csv('lassoModelChosen.ds_y_test_pred.csv', sep=',', header=True, index=False)




########## Comparing and Visualizing the results
 
print('********** Comparison of Multiple Linear Regression **********')

# Building the two lists for visualization
models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
scores = [linearModel.score(X_test, y_test),
         ridgeModelChosen.score(X_test, y_test),
         lassoModelChosen.score(X_test, y_test)]
 
# Building the dictionary to compare the scores
mapping = {}
mapping['Linear Regreesion'] = linearModel.score(X_test, y_test)
mapping['Ridge Regreesion'] = ridgeModelChosen.score(X_test, y_test)
mapping['Lasso Regression'] = lassoModelChosen.score(X_test, y_test)
 
# Printing the scores for different models
for key, val in mapping.items():
    print(str(key)+' : '+str(val))


# Plotting the scores
plt.figure(num=1, figsize=(12, 6))
plt.bar(models, scores)
plt.xlabel('Regression Models')
plt.ylabel('Score')
plt.savefig('Fig_1.png')
plt.show()
plt.close('all')