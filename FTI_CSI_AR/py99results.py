##### import libraries

import pandas as pd
#import numpy as np




##### Load all files

y          = pd.read_csv('y.csv', header=0)

ymmu       = pd.read_csv('ymmu.csv', header=0)

sigma      = pd.read_csv('sigma.csv', header=0)

FTI        = pd.read_csv('FTI.csv', header=0)

FTIdma     = pd.read_csv('FTIdma.csv', header=0)

FTIpctrank = pd.read_csv('FTIpctrank.csv', header=0)




##### Change column names

print(y.columns)
y.rename(columns = lambda x: 'y_' + x, inplace=True)
print(y.columns)
y.rename(columns = {'y_Date':'Date'}, inplace=True)
print(y.columns)


ymmu.rename(columns = lambda x: 'ymmu_' + x, inplace=True)
ymmu.rename(columns = {'ymmu_Date':'Date'}, inplace=True)


sigma.rename(columns = lambda x: 'sigma_' + x, inplace=True)
sigma.rename(columns = {'sigma_Date':'Date'}, inplace=True)


FTI.rename(columns = lambda x: 'FTI_' + x, inplace=True)
FTI.rename(columns = {'FTI_Date':'Date'}, inplace=True)


FTIdma.rename(columns = lambda x: 'FTIdma_' + x, inplace=True)
FTIdma.rename(columns = {'FTIdma_Date':'Date'}, inplace=True)


FTIpctrank.rename(columns = lambda x: 'FTIpctrank_' + x, inplace=True)
FTIpctrank.rename(columns = {'FTIpctrank_Date':'Date'}, inplace=True)



##### Merge all files

results_FTI = pd.merge(y, ymmu, on='Date', how='outer')

results_FTI = pd.merge(results_FTI, sigma, on='Date', how='outer')

results_FTI = pd.merge(results_FTI, FTI, on='Date', how='outer')

results_FTI = pd.merge(results_FTI, FTIdma, on='Date', how='outer')

results_FTI = pd.merge(results_FTI, FTIpctrank, on='Date', how='outer')




##### Save all files

results_FTI.to_csv("results_FTI.csv", index=False)