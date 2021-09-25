#################### Decomposing Time Series Data into Trend, Seasonality, and Residual : Additive Model and Multiplicative Model ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/09/25
# Last Updated: 2021/09/25
#
# Github:
# https://github.com/yoshisatoh/Stats/tree/main/Trend_and_Seasonality/ts.py
# https://github.com/yoshisatoh/Stats/blob/main/Trend_and_Seasonality/ts.py
#
#
########## Input Data File(s)
#
#airline-passengers.csv
#
#
########## Usage Instructions
#
#Run this code on Terminal of MacOS as follows:
#python3 ts.py (raw dataset csv file) (model: additive or multiplicative) (one period/cycle of seasonality in the number of data points)
#
#For instance,
#python3 ts.py airline-passengers.csv multiplicative 12
#python3 ts.py covid19.csv additive 7
#
#Additive Model
# y(t) = Level + Trend + Seasonality + Noise
#
#Multiplicative Model
#y(t) = Level * Trend * Seasonality * Noise
#
#
########## References:
#
#https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
#https://stackoverflow.com/questions/45184055/how-to-plot-multiple-seasonal-decompose-plots-in-one-figure
#https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
#
#
####################




##### import

import sys
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose




##### arguments

csvf = str(sys.argv[1])    # e.g., airline-passengers.csv
md   = str(sys.argv[2])    # e.g., multiplicative
frq  = int(sys.argv[3])    # e.g., 12




##### Plot: Raw Dataset

dt_now = datetime.datetime.now()
#print(dt_now)
# 2019-02-04 21:04:15.412854

series = pd.read_csv(csvf, header=0, index_col=0)
#print(series)
'''
         Passengers
Month              
1949-01         112
1949-02         118
1949-03         132
1949-04         129
1949-05         121
...             ...
1960-08         606
1960-09         508
1960-10         461
1960-11         390
1960-12         432
'''
#print(type(series))
#<class 'pandas.core.frame.DataFrame'>

series.plot(figsize=(12,9))

plt.title("Raw Data")

#plt.savefig("Figure_1_raw_data_" + dt_now.strftime('%Y-%m-%d_%H%M%S') + ".png")
plt.savefig("Figure_1_raw_data.png")

plt.show()




##### Plot: Decomposing Raw (Observed) Data into Trend, Seasonality, and Residual

dta = series
#print(dta)
'''
         Passengers
Month              
1949-01         112
1949-02         118
1949-03         132
1949-04         129
1949-05         121
...             ...
1960-08         606
1960-09         508
1960-10         461
1960-11         390
1960-12         432

[144 rows x 1 columns]
'''
#print(type(dta))
#<class 'pandas.core.frame.DataFrame'>

yname = dta.columns[0]

dta.eval(yname).interpolate(inplace=True)

res = seasonal_decompose(dta.eval(yname), model=md, freq=frq)
#print(res)

rescsv = pd.concat([res.observed, res.trend, res.seasonal, res.resid], axis=1, join='outer')
#print(rescsv)

pd.DataFrame(data=rescsv).to_csv("rescsv.csv", header=True, index=True)


def plotseasonal(res, axes ):
    res.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')
    res.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    res.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    res.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')




##### plot
#
# [Case A] When plotting one raw dataset and its trend, seasonality, and residual
fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(12,9))
#print(type(fig))
#<class 'matplotlib.figure.Figure'>
#
#print(type(axes))
#<class 'numpy.ndarray'>
#
#
#print(axes)
'''
[<matplotlib.axes._subplots.AxesSubplot object at 0x128497130>
 <matplotlib.axes._subplots.AxesSubplot object at 0x129abe2b0>
 <matplotlib.axes._subplots.AxesSubplot object at 0x129aea430>
 <matplotlib.axes._subplots.AxesSubplot object at 0x129b17580>]
'''
plotseasonal(res, axes[:])
#
# 
# [Case B]  When plotting three raw datasets and its trend, seasonality, and residual
#fig, axes = plt.subplots(ncols=3, nrows=4, sharex=True, figsize=(12,5))
#plotseasonal(res, axes[:,0])
#plotseasonal(res, axes[:,1])
#plotseasonal(res, axes[:,2])


plt.tight_layout()

#plt.savefig("Figure_2_raw_data_trend_seasonality_residual_" + md + "_" + dt_now.strftime('%Y-%m-%d_%H%M%S') + ".png")
plt.savefig("Figure_2_raw_data_trend_seasonality_residual.png")

plt.show()




