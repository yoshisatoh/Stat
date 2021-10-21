#################### Multiple Linear Regression (by selected two dependent variables x1 and x2) ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/09/25
# Last Updated: 2021/09/25
#
# Github:
# https://github.com/yoshisatoh/Stat/tree/main/Linear_Regression/X_raw_standardized_normalized/multi_class/no_regularization/othermlr.py
#
#
########## Input Data Files
#
#(output files from mlrrwstnmslr.py)
#Xconv_Y_Ypred_YpredSingle.csv
#clifIntercept.txt
#clifCoefficients.txt
#
#
########## Usage Instructions
#
#Multiple Linear Regression with an explained variable Y and specified explanatory variables Xs (e.g., RM, LSAT)
#python3 othermlr.py Xconv_Y_Ypred_YpredSingle.csv clifIntercept.txt clifCoefficients.txt Y RM LSTAT
#
#
####################



########## import

import sys
import pandas as pd
import matplotlib.pyplot as plt



########## arguments

dfname = str(sys.argv[1])    # e.g., Xconv_Y_Ypred_YpredSingle.csv
clifI = str(sys.argv[2])    #clifIntercept.txt
clifC = str(sys.argv[3])    #clifCoefficients.txt
yname  = str(sys.argv[4])    # e.g., Y (dependent vairible)
xname1  = str(sys.argv[5])    # e.g., RM (one of multiple indepdent variables)
xname2  = str(sys.argv[6])    # e.g., LSAT (one of multiple indepdent variables)



########## data load

df = pd.read_csv(dfname, index_col=0)
#print(df)
'''
         CRIM        ZN     INDUS      CHAS  ...     LSTAT     Y      Ypred  YpredSingle
0   -0.419782  0.284830 -1.287909 -0.272599  ... -1.075562  24.0  30.003843    23.639060
1   -0.417339 -0.487722 -0.593381 -0.272599  ... -0.492439  21.6  25.025562    23.052341
2   -0.417342 -0.487722 -0.593381 -0.272599  ... -1.208727  34.7  30.567597    25.963078
3   -0.416750 -0.487722 -1.306878 -0.272599  ... -1.361517  33.4  28.607036    25.250633
4   -0.412482 -0.487722 -1.306878 -0.272599  ... -1.026501  36.2  27.943524    25.818303
..        ...       ...       ...       ...  ...       ...   ...        ...          ...
501 -0.413229 -0.487722  0.115738 -0.272599  ... -0.418147  22.4  23.533341    23.707638
502 -0.415249 -0.487722  0.115738 -0.272599  ... -0.500850  20.6  22.375719    21.905571
503 -0.413447 -0.487722  0.115738 -0.272599  ... -0.983048  23.9  27.627426    25.166816
504 -0.407764 -0.487722  0.115738 -0.272599  ... -0.865302  22.0  26.127967    24.473421
505 -0.415000 -0.487722  0.115738 -0.272599  ... -0.669058  11.9  22.344212    21.562684

[506 rows x 16 columns]
'''


##### intercept b0

f0 = open(clifI)
b0 = f0.read()
f0.close()
#print(type(b0))
#<class 'str'>
b0 = float(b0)
#print(b0)
#22.532806324110673


##### coefficients b1 and b2
dfb = pd.read_csv(clifC, delim_whitespace=True)
#
#print(dfb)
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
#print(dfb.columns)
#Index(['Name', 'Coefficients'], dtype='object')
#
#print(dfb.columns[0])
#Name

#print(dfb.columns[1])
#Coefficients
#
#print(dfb['Name'])
#print(dfb['Coefficients'])


#print(float(dfb[dfb.Name == xname1]['Coefficients']))
#print(float(dfb[dfb.eval(dfb.columns[0]) == xname1]['Coefficients']))
#2.67423
#print(type(float(dfb[dfb.eval(dfb.columns[0]) == xname1]['Coefficients'])))
#<class 'float'>
#
#print(float(dfb[dfb.eval(dfb.columns[0]) == xname1]['Coefficients']))
#2.67423
#
#print(float(dfb[dfb.eval(dfb.columns[0]) == xname1][dfb.columns[1]]))
#2.67423
#print(
#type(
#float(dfb[dfb.eval(dfb.columns[0]) == xname1][dfb.columns[1]])
#)
#)
#<class 'float'>

b1 = float(dfb[dfb.eval(dfb.columns[0]) == xname1][dfb.columns[1]])
#print(b1)
#2.67423

b2 = float(dfb[dfb.eval(dfb.columns[0]) == xname2][dfb.columns[1]])
#print(b2)
#-3.7436269999999996


#print(df[xname1])
#print(df[xname2])


Ypred2 = b0 + b1 * df[xname1] + b2 * df[xname2]
#
#print(Ypred2)
#print(type(Ypred2))
#<class 'pandas.core.series.Series'>

Ypred2 = pd.DataFrame(Ypred2)
#print(Ypred2)
Ypred2 = Ypred2.rename(columns={0: 'Ypred2'})
#print(Ypred2)

Xconv_Y_Ypred_YpredSingle_Ypred2 = pd.concat([df, Ypred2], axis=1)
Xconv_Y_Ypred_YpredSingle_Ypred2.to_csv('Xconv_Y_Ypred_YpredSingle_Ypred2.csv', header=True, index=True)



########## Multiple Linear Regeression by using xname1 and xname2: plot xname1

plt.figure(3)

plt.title('Multiple Linear Regression: ' + str(xname1) + ' & ' + str(xname2))
plt.xlabel(xname1, fontsize=14)
plt.ylabel(yname, fontsize=14)

plt.scatter(Xconv_Y_Ypred_YpredSingle_Ypred2.eval(xname1), Xconv_Y_Ypred_YpredSingle_Ypred2.eval(yname), c='blue', label='Y & X1')
plt.scatter(Xconv_Y_Ypred_YpredSingle_Ypred2.eval(xname1), Xconv_Y_Ypred_YpredSingle_Ypred2.Ypred2, c='red', label='Ypred2 (Predicted Y) & X1')

plt.xlim(min(Xconv_Y_Ypred_YpredSingle_Ypred2.eval(xname1)), max(Xconv_Y_Ypred_YpredSingle_Ypred2.eval(xname1)))
plt.ylim(
min(min(Xconv_Y_Ypred_YpredSingle_Ypred2[yname]), min(Xconv_Y_Ypred_YpredSingle_Ypred2['Ypred']), min(Xconv_Y_Ypred_YpredSingle_Ypred2['Ypred2'])),
max(max(Xconv_Y_Ypred_YpredSingle_Ypred2[yname]), max(Xconv_Y_Ypred_YpredSingle_Ypred2['Ypred']), max(Xconv_Y_Ypred_YpredSingle_Ypred2['Ypred2']))
)
#print(Xconv_Y_Ypred_YpredSingle_Ypred2['Ypred2'])
#print(min(Xconv_Y_Ypred_YpredSingle_Ypred2['Ypred2']))
#1.0691449482732711

plt.legend(loc='lower right', fontsize=12)
#
#print(max(Y))
#print(max(Ypred['Ypred']))
#
#plt.text(min(Xconv[xname]), max(max(Y), max(Ypred['Ypred'])), 'R^2 = ' + str(round(clf.score(Xconv,Y), 4)), fontsize=10)
plt.savefig("Figure_Multiple_Linear_Regression_" + str(xname1) + "_and_" + str(xname2) + "_x-axis_" + str(xname1) + ".png")
plt.show()



########## Multiple Linear Regeression by using xname1 and xname2: plot xname2

plt.figure(4)

plt.title('Multiple Linear Regression: ' + str(xname1) + ' & ' + str(xname2))
plt.xlabel(xname2, fontsize=14)
plt.ylabel(yname, fontsize=14)

plt.scatter(Xconv_Y_Ypred_YpredSingle_Ypred2.eval(xname2), Xconv_Y_Ypred_YpredSingle_Ypred2.Y, c='blue', label='Y & X2')
plt.scatter(Xconv_Y_Ypred_YpredSingle_Ypred2.eval(xname2), Xconv_Y_Ypred_YpredSingle_Ypred2.Ypred2, c='red', label='Ypred2 (Predicted Y) & X2')

plt.xlim(min(Xconv_Y_Ypred_YpredSingle_Ypred2.eval(xname2)), max(Xconv_Y_Ypred_YpredSingle_Ypred2.eval(xname2)))
plt.ylim(
min(min(Xconv_Y_Ypred_YpredSingle_Ypred2[yname]), min(Xconv_Y_Ypred_YpredSingle_Ypred2['Ypred']), min(Xconv_Y_Ypred_YpredSingle_Ypred2['Ypred2'])),
max(max(Xconv_Y_Ypred_YpredSingle_Ypred2[yname]), max(Xconv_Y_Ypred_YpredSingle_Ypred2['Ypred']), max(Xconv_Y_Ypred_YpredSingle_Ypred2['Ypred2']))
)

plt.legend(loc='lower right', fontsize=12)
#
#print(max(Y))
#print(max(Ypred['Ypred']))
#
#plt.text(min(Xconv[xname]), max(max(Y), max(Ypred['Ypred'])), 'R^2 = ' + str(round(clf.score(Xconv,Y), 4)), fontsize=10)
plt.savefig("Figure_Multiple_Linear_Regression_" + str(xname1) + "_and_" + str(xname2) + "_x-axis_" + str(xname2) + ".png")
plt.show()
