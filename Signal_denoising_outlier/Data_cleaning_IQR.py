# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Dr. Dipankar Mandal, dmandal@ksu.edu

"""
# ----------------------------------------------------------------------------
  # Copyright (C) 2022 by PrecisionAG, Agronomy, KSU
 
  # This program is free software; you can redistribute it and/or modify it
  # under the terms of the GNU General Public License as published by the Free
  # Software Foundation; either version 3 of the License, or (at your option)
  # any later version.
  # This program is distributed in the hope that it will be useful, but WITHOUT
  # ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  # FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
  # more details.
 
  # You should have received a copy of the GNU General Public License along
  # with this program; if not, see http://www.gnu.org/licenses/
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12,2]

###-----------------------------------------------------------------------------
ss = '1'

filename = 'ardec0710'

## Reading TDR-----------------------------------------------------------------
dfo = pd.read_excel(filename+'_excel_Plot'+ss+'_WTdenoised.xlsx', 
                   na_values = ['no info', '.','None','#VALUE!', '#DIV/0!','NA',''])

del dfo['Unnamed: 0']


## Data cleaning
##-----------------------------------------------------------------------------
## A threshold at 20mV is set, and the readings for which the FRF_R had a 
## value below this threshold are discarded from the data set. 
df2 = dfo[dfo.FRF_R > 20]
n = df2.shape[0]

## Fluorescence indices calculation--------------------------------------------
df2['NBI_R'] = df2['FRF_UV']/df2['FRF_R']
df2['NBI_G'] = df2['FRF_UV']/df2['FRF_G']
df2['NBI_B'] = df2['FRF_UV']/df2['RF_B']
df2['NBI1'] = (df2['FRF_UV']+df2['FRF_G'])/(df2['FRF_R']**2)
df2['CHL'] = df2['FRF_R']/df2['RF_R']
df2['CHL1'] = df2['FRF_G']/df2['RF_R']
df2['FLAV'] = np.log10(df2['FRF_R']/df2['FRF_UV'])
#df2.to_excel(filename+'_Plot'+str(ss)+'_WTdenoised_index.xlsx')


## Outliers removal using the Inter Quartile Range (IQR) method---------------- 
s = 'NBI_R'
Q1 = df2[s].quantile(0.25)
Q3 = df2[s].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 
filter = (df2[s] >= Q1 - 1.5 * IQR) & (df2[s] <= Q3 + 1.5 *IQR)
df2f = df2.loc[filter] 
df2f_mean = df2f[s].mean()
df2f_std = df2f[s].std()
df2f[s].to_excel(filename+'_Plot'+str(ss)+'_WTdenoised_'+s+'.xlsx')
NBI_R = df2f_mean
NBI_R_std = df2f_std


s = 'NBI_G'
Q1 = df2[s].quantile(0.25)
Q3 = df2[s].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 
filter = (df2[s] >= Q1 - 1.5 * IQR) & (df2[s] <= Q3 + 1.5 *IQR)
df2f = df2.loc[filter] 
df2f_mean = df2f[s].mean()
df2f_std = df2f[s].std()
df2f[s].to_excel(filename+'_Plot'+str(ss)+'_WTdenoised_'+s+'.xlsx')
NBI_G = df2f_mean
NBI_G_std = df2f_std


s = 'NBI_B'
Q1 = df2[s].quantile(0.25)
Q3 = df2[s].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 
filter = (df2[s] >= Q1 - 1.5 * IQR) & (df2[s] <= Q3 + 1.5 *IQR)
df2f = df2.loc[filter] 
df2f_mean = df2f[s].mean()
df2f_std = df2f[s].std()
df2f[s].to_excel(filename+'_Plot'+str(ss)+'_WTdenoised_'+s+'.xlsx')
NBI_B = df2f_mean
NBI_B_std = df2f_std


s = 'NBI1'
Q1 = df2[s].quantile(0.25)
Q3 = df2[s].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 
filter = (df2[s] >= Q1 - 1.5 * IQR) & (df2[s] <= Q3 + 1.5 *IQR)
df2f = df2.loc[filter] 
df2f_mean = df2f[s].mean()
df2f_std = df2f[s].std()
df2f[s].to_excel(filename+'_Plot'+str(ss)+'_WTdenoised_'+s+'.xlsx')
NBI1 = df2f_mean
NBI1_std = df2f_std


s = 'CHL'
Q1 = df2[s].quantile(0.25)
Q3 = df2[s].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 
filter = (df2[s] >= Q1 - 1.5 * IQR) & (df2[s] <= Q3 + 1.5 *IQR)
df2f = df2.loc[filter] 
df2f_mean = df2f[s].mean()
df2f_std = df2f[s].std()
df2f[s].to_excel(filename+'_Plot'+str(ss)+'_WTdenoised_'+s+'.xlsx')
CHL = df2f_mean
CHL_std = df2f_std


s = 'CHL1'
Q1 = df2[s].quantile(0.25)
Q3 = df2[s].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 
filter = (df2[s] >= Q1 - 1.5 * IQR) & (df2[s] <= Q3 + 1.5 *IQR)
df2f = df2.loc[filter] 
df2f_mean = df2f[s].mean()
df2f_std = df2f[s].std()
df2f[s].to_excel(filename+'_Plot'+str(ss)+'_WTdenoised_'+s+'.xlsx')
CHL1 = df2f_mean
CHL1_std = df2f_std


s = 'FLAV'
Q1 = df2[s].quantile(0.25)
Q3 = df2[s].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 
filter = (df2[s] >= Q1 - 1.5 * IQR) & (df2[s] <= Q3 + 1.5 *IQR)
df2f = df2.loc[filter] 
df2f_mean = df2f[s].mean()
df2f_std = df2f[s].std()
df2f[s].to_excel(filename+'_Plot'+str(ss)+'_WTdenoised_'+s+'.xlsx')
FLAV = df2f_mean
FLAV_std = df2f_std


##============================================================================
dfObj = pd.DataFrame()
dfObj['Indices'] = ['NBI_R', 'NBI_G', 'NBI_B', 'NBI1', 'CHL', 'CHL1', 'FLAV','NBI_R_std', 'NBI_G_std', 'NBI_B_std', 'NBI1_std', 'CHL_std', 'CHL1_std', 'FLAV_std']
dfObj['Values'] = [NBI_R, NBI_G, NBI_B, NBI1, CHL, CHL1, FLAV, NBI_R_std, NBI_G_std, NBI_B_std, NBI1_std, CHL_std, CHL1_std, FLAV_std]
dfObj = dfObj.T
## Export it as an excel file
dfObj.to_excel(filename+'_Plot'+str(ss)+'_WTdenoised_sumindex.xlsx')
