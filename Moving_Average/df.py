#################### Getting COVID-19 data via the Internet ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/02
# Last Updated: 2021/10/02
#
# Github:
# https://github.com/yoshisatoh/Stat/tree/main/moving_average/df.py
# https://github.com/yoshisatoh/Stat/blob/main/moving_average/df.py
#
#
########## Input Data File(s)
#
#N/A
#This program gets data via the Internet.
#
#
########## Usage Instructions
#
# Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python df.py (iso_code) (a specified data column for x-axis) (a specified data column for y-axis)
#
# For instance,
#python df.py JPN date new_cases
#
#
########## Output Data File(s)
#
#'isocodef.txt'
#(iso_code) specified as a first argument will be recorded in this txt file.
#
#df.csv    # entire raw data
#dfiso1.csv    #iso_code,location,date,new_cases or generally iso_code,location,(x-axis),(y-axis)
#dfiso2.csv    #date, new_cases or generally (x-axis),(y-axis)
#
#
########## Data Sources
#
#https://ourworldindata.org/coronavirus-source-data
#https://covid.ourworldindata.org/data/owid-covid-data.csv
#
#
####################



########## import Python libraries

import pandas as pd
import io
import requests
import sys

#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context




########## arguments

isocode = str(sys.argv[1])
#isocode = str("JPN")

isocodef = 'isocodef.txt'
s = isocode

with open(isocodef, mode='w') as f:
    f.write(s)

# specified data columns
xname = str(sys.argv[2])    #date
yname = str(sys.argv[3])    #new_cases




########## data source

url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

#dfurl = requests.get(url).content
dfurl = requests.get(url, verify=False).content

#df = pd.read_csv(dfurl)
df = pd.read_csv(io.StringIO(dfurl.decode('utf-8')))
pd.DataFrame(data=df).to_csv("df.csv", header=True, index=False)

#print(df.columns)
'''
Index(['iso_code', 'location', 'date', 'total_cases', 'new_cases',
       'total_deaths', 'new_deaths', 'total_cases_per_million',
       'new_cases_per_million', 'total_deaths_per_million',
       'new_deaths_per_million', 'total_tests', 'new_tests',
       'total_tests_per_thousand', 'new_tests_per_thousand',
       'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_units',
       'stringency_index', 'population', 'population_density', 'median_age',
       'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'extreme_poverty',
       'cvd_death_rate', 'diabetes_prevalence', 'female_smokers',
       'male_smokers', 'handwashing_facilities', 'hospital_beds_per_100k'],
      dtype='object')
'''


#dfiso1 = df[df['iso_code'] == isocode][['iso_code', 'location', 'date', 'new_cases']]
dfiso1 = df[df['iso_code'] == isocode][['iso_code', 'location', xname, yname]]

pd.DataFrame(data=dfiso1).to_csv("dfiso1.csv", header=True, index=False)

#dfiso2 = dfiso1[['date', 'new_cases']]
dfiso2 = dfiso1[[xname, yname]]
pd.DataFrame(data=dfiso2).to_csv("dfiso2.csv", header=True, index=False)