##### import quandl library
import quandl

##### import csv library to write and read csv files
import csv

##### import pandas library
import pandas as pd


### TO UPDATE
#quandl.ApiConfig.api_key = "YOUR_KEY_HERE"
quandl.ApiConfig.api_key = "dgqzKwg2867aCQAfCbCs"


#####Quandl data
# go to: https://www.quandl.com/search
#
# Filters
# - Free
#
# Asset Class
# - Equities
#
# Data Type
# - Price & Volumes
#
# Region
# - United States
#
# Select:
#NASDAQ OMX Global Index Data
#https://www.quandl.com/data/NASDAQOMX-NASDAQ-OMX-Global-Index-Data
#
# Then see each page of the follwoing three indices:
#NASDAQ-100 (NDX)
#NASDAQ Transportation (TRAN)
#NASDAQ Financial 100 (IXF)


#NASDAQ-100 (NDX)
#https://www.quandl.com/data/NASDAQOMX/NDX-NASDAQ-100-NDX
# Press EXPORT DATA > Libraries: Python
# Copy

def NDX():
    dt_NDX = quandl.get("NASDAQOMX/NDX")
    dt_NDX['NDX'] = dt_NDX['Index Value'].pct_change()
    dt_NDX['NDX'].to_csv("NDX.csv")

NDX()


#NASDAQ Transportation (TRAN)
#https://www.quandl.com/data/NASDAQOMX/TRAN-NASDAQ-Transportation-TRAN
# Press EXPORT DATA > Libraries: Python
# Copy

def TRAN():
    dt_TRAN = quandl.get("NASDAQOMX/TRAN")
    dt_TRAN['TRAN'] = dt_TRAN['Index Value'].pct_change()
    dt_TRAN['TRAN'].to_csv("TRAN.csv")

TRAN()


#NASDAQ Financial 100 (IXF)
#https://www.quandl.com/data/NASDAQOMX/IXF-NASDAQ-Financial-100-IXF
# Press EXPORT DATA > Libraries: Python
# Copy

def IXF():
    dt_IXF = quandl.get("NASDAQOMX/IXF")
    dt_IXF['IXF'] = dt_IXF['Index Value'].pct_change()
    dt_IXF['IXF'].to_csv("IXF.csv")

IXF()



##### merge several y data into one by using pandas.concat

NDX = pd.read_csv('NDX.csv', header=0)
NDX = NDX.set_index('Trade Date')
print(NDX)

TRAN = pd.read_csv('TRAN.csv', header=0)
TRAN = TRAN.set_index('Trade Date')
print(TRAN)

IXF = pd.read_csv('IXF.csv', header=0)
IXF = IXF.set_index('Trade Date')
print(IXF)


y = pd.concat([NDX, TRAN, IXF], axis=1, join='inner')
print(y)
y = y.dropna(how="any")
print(y)
y.to_csv("y.csv")