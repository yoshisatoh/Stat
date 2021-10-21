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
# https://github.com/yoshisatoh/Stat/tree/main/Linear_Regression/X_raw_standardized_normalized/multi_class/no_regularization/mlrrwstnmslr.py
#
#
########## Input Data Files
#
#df.csv
#
#
########## Usage Instructions
#
#python3 mlrrwstnm.py (csv data file) (dependent variable Y) (specified single independent variable x) (x conversion)
#
#For instance,
#
#[Option A] Raw Data X vs Raw Data Y
#python3 mlrrwstnmslr.py df.csv Y RM r
#
#[Option B] Standardized Data X vs Raw Data Y
#python3 mlrrwstnmslr.py df.csv Y RM s
#
#[Option C] Normalized Data X vs Raw Data Y
#python3 mlrrwstnmslr.py df.csv Y RM n
#
#
####################




########## import

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm


########## arguments

dfname = str(sys.argv[1])    # e.g., df.csv
yname  = str(sys.argv[2])    # e.g., Y (dependent vairible)
xname  = str(sys.argv[3])    # e.g., RM (one of multiple indepdent variables to show)
xconv  = str(sys.argv[4])    # e.g., r (raw X data), s (standardized X data), or n (normalized X data)



########### load csv dataset

##dt = datasets.load_boston()
##
## Boston house price dataset dict-keys
##
##print(type(dt))
##<class 'sklearn.utils.Bunch'>
##
##print(dt.keys())
##dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])
##
##print(pd.DataFrame(dt.data, columns=boston.feature_names))
'''
        CRIM    ZN  INDUS  CHAS    NOX  ...  RAD    TAX  PTRATIO       B  LSTAT
0    0.00632  18.0   2.31   0.0  0.538  ...  1.0  296.0     15.3  396.90   4.98
1    0.02731   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  396.90   9.14
2    0.02729   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  392.83   4.03
3    0.03237   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  394.63   2.94
4    0.06905   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  396.90   5.33
..       ...   ...    ...   ...    ...  ...  ...    ...      ...     ...    ...
501  0.06263   0.0  11.93   0.0  0.573  ...  1.0  273.0     21.0  391.99   9.67
502  0.04527   0.0  11.93   0.0  0.573  ...  1.0  273.0     21.0  396.90   9.08
503  0.06076   0.0  11.93   0.0  0.573  ...  1.0  273.0     21.0  396.90   5.64
504  0.10959   0.0  11.93   0.0  0.573  ...  1.0  273.0     21.0  393.45   6.48
505  0.04741   0.0  11.93   0.0  0.573  ...  1.0  273.0     21.0  396.90   7.88

[506 rows x 13 columns]
'''
##
'''
CRIM    crime rate
ZN    density of residential area
INDUS    rate of industrial area (non-retail)
CHAS    Charles River (1: around the river, 0: others)
NOX    NOx concentration
RM    average number of rooms par house
AGE    rate of houses built before 1940
DIS     weighted distance from five employers in Boston
RAD    accessibility to major roads
TAX    income tax (rates) par $10,000
PTRATIO    number of students par teacher
B    rate of African Americans 1000(Bk – 0.63)^2
LSTAT    rate of low income earners
'''
##
##df = pd.DataFrame(dt.data, columns=dt.feature_names)    #data: independent variables
##df['Y'] = dt.target    # Y: target/dependent variable is added
###df.to_csv('df.csv', header=True, index=True)

df = pd.read_csv(dfname, index_col=0)
#
#print(df)
#[506 rows x 14 columns]
#
#print(df.columns)
'''
Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT', 'Y'],
      dtype='object')
'''



########### set Y (dependent variable)
Y = df.eval(yname)
#
#print(Y)
'''
0      24.0
1      21.6
2      34.7
3      33.4
4      36.2
       ... 
501    22.4
502    20.6
503    23.9
504    22.0
505    11.9
Name: Y, Length: 506, dtype: float64
'''



########### set X (independent variables)
if xconv == 'r':
    #r: raw X data
    print('r: raw X data')
    #
    ##### Raw data X
    #
    #print("########## Raw Data X ##########")
    X = df.drop(yname, axis=1)
    Xconv = X    # no conversion, but to have the same variable, call X as Xconv
    #
    with open('Xconv.describe.txt', 'w') as f:
        print(Xconv.describe(), file=f)
    
    
elif xconv == 's':
    X = df.drop(yname, axis=1)
    #
    #s: standardized X data
    print('s: standardized X data')
    #
    # Standardized data X (average == 0, standard deviation == 1 for all x)
    #
    scaler = StandardScaler()
    scaler.fit(X)
    #
    #print("########## Standardized Data X ##########")
    #print("scaler.fit(X) ->" + str(scaler.fit(X)))
    #print("scaler.mean_ ->" + str(scaler.mean_))
    #print("scaler.transform ->" + str(scaler.transform(X)))
    #
    Xconv = scaler.transform(X)
    Xconv = pd.DataFrame(Xconv)
    #Xconv.columns = boston.feature_names
    Xconv.columns = X.columns
    #
    #print(type(Xconv.describe()))
    #<class 'pandas.core.frame.DataFrame'>
    #
    #print(Xconv.describe())
    '''
                 CRIM          ZN       INDUS        CHAS  ...         TAX     PTRATIO           B       LSTAT
    count  506.000000  506.000000  506.000000  506.000000  ...  506.000000  506.000000  506.000000  506.000000
    mean     3.613524   11.363636   11.136779    0.069170  ...  408.237154   18.455534  356.674032   12.653063
    std      8.601545   23.322453    6.860353    0.253994  ...  168.537116    2.164946   91.294864    7.141062
    min      0.006320    0.000000    0.460000    0.000000  ...  187.000000   12.600000    0.320000    1.730000
    25%      0.082045    0.000000    5.190000    0.000000  ...  279.000000   17.400000  375.377500    6.950000
    50%      0.256510    0.000000    9.690000    0.000000  ...  330.000000   19.050000  391.440000   11.360000
    75%      3.677082   12.500000   18.100000    0.000000  ...  666.000000   20.200000  396.225000   16.955000
    max     88.976200  100.000000   27.740000    1.000000  ...  711.000000   22.000000  396.900000   37.970000
    
    [8 rows x 13 columns]
    '''
    #
    #print(Xconv.describe().columns)
    '''
    Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT'],
      dtype='object')
    '''
    #
    #print(Xconv.describe()[xname])
    '''
    count    5.060000e+02
    mean    -4.563763e-17
    std      1.000990e+00
    min     -3.880249e+00
    25%     -5.686303e-01
    50%     -1.084655e-01
    75%      4.827678e-01
    max      3.555044e+00
    Name: RM, dtype: float64
    '''
    #
    #print(Xconv.describe()[xname].loc['mean'])
    #-4.563762630870209e-17
    #
    with open('Xconv.describe.txt', 'w') as f:
        print(Xconv.describe(), file=f)
    
    
elif xconv == 'n':
    X = df.drop(yname, axis=1)
    #
    #n: normalized X data
    print('n: normalized X data')
    #
    # Normalized data X (all x are normalized to take values from 0 to 1)
    mscaler = MinMaxScaler()
    mscaler.fit(X)
    #
    #print("########## Normalized Data X ##########")
    #print("mscaler.fit(X) ->" + str(mscaler.fit(X)))
    #print("mscaler.data_max_ ->" + str(mscaler.data_max_))
    #print("mscaler.data_range_ ->" + str(mscaler.data_range_))
    #print("mscaler.transform ->" + str(mscaler.transform(X)))
    #
    Xconv = mscaler.transform(X)
    Xconv = pd.DataFrame(Xconv)
    #Xconv.columns = boston.feature_names
    Xconv.columns = X.columns
    #
    with open('Xconv.describe.txt', 'w') as f:
        print(Xconv.describe(), file=f)
    
    
else:
    print('Speficy X data conversion: r (raw X data), s (standardized X data), or n (normalized X data)')
    sys.exit(1)    #error status = 1



########## Multiple Linear Regression

clf = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
clf.fit(Xconv, Y)
#SGDRegressor
#clf_SGD = linear_model.SGDRegressor(max_iter=500)
#clf_SGD.fit(Xconv, Y)
#
#
smX = sm.add_constant(Xconv)
estX = sm.OLS(Y, smX)
estXfit = estX.fit()
#
#print(estXfit.summary())
#print(type(estXfit.summary()))
#<class 'statsmodels.iolib.summary.Summary'>
with open('estXfit.summary.txt', 'w') as f:
  print(estXfit.summary(), file=f)

Yprednp = clf.predict(Xconv)    # predicted Y by using all Xs
#print(type(Yprednp))
#<class 'numpy.ndarray'>
Ypred = pd.DataFrame(Yprednp)
Ypred = Ypred.rename(columns={0: 'Ypred'})
#print(type(Ypred))
#<class 'pandas.core.frame.DataFrame'>



########## print out

#print(pd.DataFrame({"Name":Xconv.columns,
#                    "Coefficients":clf.coef_}).sort_values(by='Coefficients') ) # sorting by coefficients
#print('intercept = ' + str(clf.intercept_)) # intercept
#print('R^2 = ' + str(clf.score(Xconv,Y))) # R^2

with open('clifCoefficients.txt', 'w') as f:
  print(pd.DataFrame({"Name":Xconv.columns,
                    "Coefficients":clf.coef_}).sort_values(by='Coefficients') , file=f)

with open('clifIntercept.txt', 'w') as f:
  print(str(clf.intercept_), file=f)

with open('clifR2.txt', 'w') as f:
  print(str(clf.score(Xconv,Y)), file=f)



########## plot

plt.figure(1)

plt.title('Multiple Linear Regression: All Xs')
plt.xlabel(xname, fontsize=14)
plt.ylabel(yname, fontsize=14)

if xconv == 'r':
    plt.scatter(Xconv[xname], Y, c='blue', label='Y & Raw Data X')
elif xconv == 's':
    plt.scatter(Xconv[xname], Y, c='blue', label='Y & Standardized Data X')
elif xconv == 'n':
    plt.scatter(Xconv[xname], Y, c='blue', label='Y & Normalized Data X')
else:
    #This section is not executed since exit is done in the previous section.
    print('Speficy X data conversion: r (raw X data), s (standardized X data), or n (normalized X data)')
    sys.exit(1)    #error status = 1

#plt.scatter(Xconv.eval(xname), clf.predict(Xconv), c='red', label='Multiple Regression Analysis')
plt.scatter(Xconv[xname], Ypred, c='red', label='Ypred(Predicted Y) & X')
plt.xlim(min(Xconv[xname]), max(Xconv[xname]))
plt.ylim(
min(min(Y), min(Ypred['Ypred'])),
max(max(Y), max(Ypred['Ypred']))
)
plt.legend(loc='lower right', fontsize=12)
#
#print(max(Y))
#print(max(Ypred['Ypred']))
#
plt.text(min(Xconv[xname]), max(max(Y), max(Ypred['Ypred']))*0.90, 'R^2 = ' + str(round(clf.score(Xconv,Y), 4)), fontsize=10)
plt.savefig("Figure_Multiple_Linear_Regression_" + xconv + ".png")
plt.show()



########## output data files

#print(X)
#
#print(type(X))
#<class 'pandas.core.frame.DataFrame'>

#print(Xconv)
#
#print(type(Xconv))
#<class 'pandas.core.frame.DataFrame'>

#print(Y)
#print(type(Y))
#<class 'pandas.core.series.Series'>

X_Y = pd.concat([X, Y], axis=1)
Xconv_Y = pd.concat([Xconv, Y], axis=1)
Xconv_Y_Ypred = pd.concat([Xconv_Y, Ypred], axis=1)

X_Y.to_csv('X_Y.csv', header=True, index=True)
Xconv_Y.to_csv('Xconv_Y.csv', header=True, index=True)
Xconv_Y_Ypred.to_csv('Xconv_Y_Ypred.csv', header=True, index=True)




########## Single Linear Regeression by using xname: data generation

dfcoef = pd.DataFrame({"Name":Xconv.columns, "Coefficients":clf.coef_}).sort_values(by='Coefficients')
#
#print(dfcoef)
'''
       Name  Coefficients
12    LSTAT     -3.743627
7       DIS     -3.104044
9       TAX     -2.076782
10  PTRATIO     -2.060607
4       NOX     -2.056718
0      CRIM     -0.928146
6       AGE      0.019466
2     INDUS      0.140900
3      CHAS      0.681740
11        B      0.849268
1        ZN      1.081569
8       RAD      2.662218
5        RM      2.674230
'''
#
#print(dfcoef[dfcoef.Name == xname]['Coefficients'])
#  Name  Coefficients
#5   RM       2.67423
#
#print(dfcoef[dfcoef.Name == xname]['Coefficients'])
#5    2.67423
#Name: Coefficients, dtype: float64
#
#print(type(dfcoef[dfcoef.Name == xname]['Coefficients']))
#<class 'pandas.core.series.Series'>
#
#print(float(dfcoef[dfcoef.Name == xname]['Coefficients']))
#2.6742301652393117
#
#print('intercept = ' + str(clf.intercept_))
#intercept = 22.532806324110673
#
#print('R^2 = ' + str(clf.score(Xconv,Y)))
#R^2 = 0.7406426641094094

#print(Xconv_Y_Ypred[xname])
'''
0      0.413672
1      0.194274
2      1.282714
3      1.016303
4      1.228577
         ...   
501    0.439316
502   -0.234548
503    0.984960
504    0.725672
505   -0.362767
'''

YpredSingle = clf.intercept_ + float(dfcoef[dfcoef.Name == xname]['Coefficients']) * Xconv_Y_Ypred[xname]
#print(YpredSingle)
#print(type(YpredSingle))
#<class 'pandas.core.series.Series'>
#
#
YpredSingle = pd.DataFrame(YpredSingle)
#print(type(YpredSingle))
#<class 'pandas.core.frame.DataFrame'>
#
#print(YpredSingle.columns)
#Index(['RM'], dtype='object')
#
#YpredSingle = YpredSingle.rename(columns={xname: 'YpredSingle_' + str(xname)})
YpredSingle = YpredSingle.rename(columns={xname: 'YpredSingle'})
#print(YpredSingle.columns)
#Index(['YpredSingle'], dtype='object')

Xconv_Y_Ypred_YpredSingle = pd.concat([Xconv_Y_Ypred, YpredSingle], axis=1)

Xconv_Y_Ypred_YpredSingle.to_csv('Xconv_Y_Ypred_YpredSingle.csv', header=True, index=True)



########## Single Linear Regeression by using xname: plot

plt.figure(2)

plt.title('Single Linear Regression')
plt.xlabel(xname, fontsize=14)
plt.ylabel(yname, fontsize=14)

if xconv == 'r':
    plt.scatter(Xconv[xname], Y, c='blue', label='Y & Raw Data X')
elif xconv == 's':
    plt.scatter(Xconv[xname], Y, c='blue', label='Y & Standardized Data X')
elif xconv == 'n':
    plt.scatter(Xconv[xname], Y, c='blue', label='Y & Normalized Data X')
else:
    #This section is not executed since exit is done in the previous section.
    print('Speficy X data conversion: r (raw X data), s (standardized X data), or n (normalized X data)')
    sys.exit(1)    #error status = 1

#plt.scatter(Xconv.eval(xname), clf.predict(Xconv), c='red', label='Multiple Regression Analysis')
plt.scatter(Xconv_Y_Ypred_YpredSingle.eval(xname), Xconv_Y_Ypred_YpredSingle.YpredSingle, c='red', label='YpredSingle (Predicted Y) & X')
plt.xlim(min(Xconv_Y_Ypred_YpredSingle.eval(xname)), max(Xconv_Y_Ypred_YpredSingle.eval(xname)))
plt.ylim(
min(min(Y), min(Ypred['Ypred']), min(YpredSingle['YpredSingle'])),
max(max(Y), max(Ypred['Ypred']), max(YpredSingle['YpredSingle']))
)
plt.legend(loc='lower right', fontsize=12)
#
#print(max(Y))
#print(max(Ypred['Ypred']))
#
#plt.text(min(Xconv[xname]), max(max(Y), max(Ypred['Ypred'])), 'R^2 = ' + str(round(clf.score(Xconv,Y), 4)), fontsize=10)
plt.savefig("Figure_Single_Linear_Regression_" + xconv + ".png")
plt.show()
