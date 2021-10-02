#################### Moving Average Calculation ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/02
# Last Updated: 2021/10/02
#
# Github:
# https://github.com/yoshisatoh/Stat/tree/main/moving_average/ma.py
# https://github.com/yoshisatoh/Stat/blob/main/moving_average/ma.py
#
#
########## Input Data File(s)
#
#dfiso2.csv
#isocodef.txt
#These files are created by df.py, which is a preceding Python script.
#
#
########## Usage Instructions
#
#python ma.py (a file name that has two columns, x & y) (x-axis) (y-axis) (title text file including iso_code) (moving average window n)
#
# For instance,
#python ma.py dfiso2.csv date new_cases isocodef.txt 7
#
#
########## Output Data File(s)
#
#df2.csv    #date, new_cases(raw data)
#df2ma.csv    #date, new_cases(moving average data, n-data point average of x)
#
#
########## Data Sources
#
#https://ourworldindata.org/coronavirus-source-data
#https://covid.ourworldindata.org/data/owid-covid-data.csv
#
#
####################




########## import Python libraries ##########

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import sys
import datetime

dt_now = datetime.datetime.now()
#print(dt_now)
# 2019-02-04 21:04:15.412854
#
#print(type(dt_now))
# <class 'datetime.datetime'>
#
#print(dt_now.strftime('%Y-%m-%d_%H:%M:%S'))
# 2019-02-04_21:04:15




########## arguments ##########

dffile = str(sys.argv[1])    #dfiso2.csv

xname = str(sys.argv[2])    #date

yname = str(sys.argv[3])    #new_cases

#isocodef = 'isocodef.txt'
isocodef = str(sys.argv[4])    #isocodef.txt

xmawin = int(sys.argv[5])    #7




########## load data files ##########

df = pd.read_csv(dffile)
#
#print(df.loc[0, ['iso_code']])
#iso_code    JPN
#
#print(df.loc[0, ['iso_code']][0])
#JPN
#
#isocode = df.loc[0, ['iso_code']][0]
#print(isocode)
#JPN

with open(isocodef) as f:
    isocode = f.read()

print(isocode)




########## data conversion ##########

df2 = df[[xname, yname]]

df2[xname] = pd.to_datetime(df2[xname], format='%Y-%m-%d')

df2.set_index(xname, inplace=True)
#print(df2.loc[:, [yname]])
df2.to_csv('df2.csv')




########## plot (raw data) ##########

#fig = plt.figure()
fig = plt.figure(figsize=(16, 8))

plt.rcParams["font.size"] = 10

ax = fig.add_subplot(1,1,1)

ax.bar(df2.index, df2[yname],  color='black', label = str(yname))    # raw data


#days = dates.DayLocator()
#days = dates.DayLocator(bymonthday = None, interval = 7, tz = None)
days = dates.DayLocator(bymonthday = None, interval = xmawin, tz = None)
#
ax.xaxis.set_major_locator(days)
#
#daysFmt = dates.DateFormatter('%Y-%m-%d')
daysFmt = dates.DateFormatter('%m-%d')
#
ax.xaxis.set_major_formatter(daysFmt)

plt.xlabel(xname)
plt.ylabel(yname)
plt.title(isocode)
plt.grid()
plt.legend(bbox_to_anchor = (0, 1), loc = 'upper left', borderaxespad = 0, fontsize = 12)
#plt.legend()

plt.savefig("Figure_1_bar_chart_raw_data_" + str(isocode) + "_" + dt_now.strftime('%Y-%m-%d_%H%M%S') + ".png")
plt.show()




########## generate moving average data ##########

df2ma = df2[yname].rolling(window=xmawin).mean()

df2ma.to_csv('df2ma.csv')




########## plot (raw data and moving average data) ##########

fig = plt.figure(figsize=(16, 8))

plt.rcParams["font.size"] = 10

ax = fig.add_subplot(1,1,1)

ax.bar(df2.index, df2[yname], color='lightgray', label = str(yname))    # raw data


#ax.plot(df2ma, label = str(yname)  + ": " + str(xmawin) + "-calendar-day moving average")    # moving average data
ax.plot(df2ma, color='red', linestyle='solid', label = str(yname)  + ": " + str(xmawin) + "-calendar-day moving average")    # moving average data

#days = dates.DayLocator()
# interval of x-axis is set to 7
#days = dates.DayLocator(bymonthday = None, interval = 7, tz = None)
days = dates.DayLocator(bymonthday = None, interval = xmawin, tz = None)
#
ax.xaxis.set_major_locator(days)
#
#daysFmt = dates.DateFormatter('%Y-%m-%d')
daysFmt = dates.DateFormatter('%m-%d')
#
ax.xaxis.set_major_formatter(daysFmt)

plt.xlabel(xname)
plt.ylabel(yname)
plt.title(isocode)
plt.grid()
plt.legend(bbox_to_anchor = (0, 1), loc = 'upper left', borderaxespad = 0, fontsize = 12)
#plt.legend()

plt.savefig("Figure_2_bar_chart_raw_data_and_line_chart_moving_average_" + str(xmawin) + "_days_" + str(isocode) + "_" + dt_now.strftime('%Y-%m-%d_%H%M%S') + ".png")
plt.show()

