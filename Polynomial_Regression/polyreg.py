#################### Unsupervised Learning: Principal Component Analysis (PCA) for Dimension Reduction ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/02
# Last Updated: 2021/10/02
#
# Github:
# https://github.com/yoshisatoh/Stat/tree/main/Polynomial_Regression/polyreg.py
# https://github.com/yoshisatoh/Stat/blob/main/Polynomial_Regression/polyreg.py
#
#
########## Input Data File(s)
#
#training.csv
#test.csv
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
# When deg = 5, degrees of polynominal regression is 5,
#python polyreg.py 5 training.csv test.csv
#
#
#Generally,
#python polyreg.py (deg: degree of polynominal regression) (training data) (test data)
#
#
########## References
#
#
#
####################




########## import Python libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import sys
import os

import sympy as sym
from sympy.plotting import plot

from IPython.display import display

import pandas as pd




########## arguments

deg         = int(sys.argv[1])
trainingcsv = sys.argv[2]
testcsv     = sys.argv[3]




########## Figure preparation

plt.figure(num=1, figsize=(15, 8)) 




########## Training Data
#
#x_training = [9, 28, 38, 58, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 278, 288, 298]
#y_training = [51, 80, 112, 294, 286, 110, 59, 70, 56, 70, 104, 59, 59, 72, 87, 99, 64, 60, 74, 151, 157, 57, 83]
#

tmp_training = pd.read_csv(trainingcsv)

x_training = tmp_training.iloc[:,0].values.tolist()
y_training = tmp_training.iloc[:,1].values.tolist()

plt.scatter(x_training, y_training, c="red", label='Training Data')
plt.grid()




########## Test Data
#
#x_test = [9, 98, 148, 198, 278]
#y_test = [56, 99, 114, 89, 173]
#

tmp_test = pd.read_csv(testcsv)

x_test = tmp_test.iloc[:,0].values.tolist()
y_test = tmp_test.iloc[:,1].values.tolist()

plt.scatter(x_test, y_test, c="blue", label='Test Data')




########## Polynominal Regression

# equally spaced, 100 points from min(x_training) to max(x_training) for ponlynominal regression models
x_latent = np.linspace(min(x_training), max(x_training), 100)


#least-squares method (Degree of the polynomial fitting: n)
#LSM (Deg: n)
cf = ["LSM (Deg: " + str(deg) +  ")", lambda x, y: np.polyfit(x, y, deg)]


sym.init_printing(use_unicode=True)
x, y = sym.symbols("x y")


#for method_name, method in [cf1, cf2, cf3, cf4, cf5, cf6, cf7, cf8, cf9]:
for method_name, method in [cf]:
    #
    print(method_name)
    #
    #
    ### calculating coefficients
    coefficients = method(x_training, y_training)
    #
    #
    ### sympy to show an equation
    expr = 0
    for index, coefficient in enumerate(coefficients):
        expr += coefficient * x ** (len(coefficients) - index - 1)
    display(sym.Eq(y, expr))
    #
    #
    ###R2
    fitted_curve = np.poly1d(method(x_training, y_training))
    #print(fitted_curve)
    r2 = r2_score(y_training, fitted_curve(x_training))
    print(r2_score(y_training, fitted_curve(x_training)))
    #0.14298353303806732
    #
    #
    ### data plotting and drawing a fitted model
    fitted_curve = np.poly1d(method(x_training, y_training))(x_latent)
    #
    ###plt.scatter(x_training, y_training, label="Training Data")
    plt.plot(x_latent, fitted_curve, c="red", label="Polynominal Regession Fitting with Training Data")
    plt.xlabel('x');plt.ylabel('y');
    plt.grid()
    #plt.legend()
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=10)
    plt.text(min(x_training),max(y_training)*0.85,sym.Eq(y, expr),fontsize=10)
    plt.text(min(x_training),max(y_training)*0.80,"R2 = " + str(r2),fontsize=10)
    plt.savefig("Figure_deg_" + str(deg) + ".png")    # added to save a figure
    plt.show()


