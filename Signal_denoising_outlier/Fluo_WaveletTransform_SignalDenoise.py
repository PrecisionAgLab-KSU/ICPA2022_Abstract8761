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
#import os, sys
import numpy as np
import pywt
## Installing Pywavelet library--  https://github.com/PyWavelets/pywt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12,5]
plt.rcParams.update({'font.size': 12})


## Input file name without extension
filename = 'ardec0710_excel'

## Lookinto the extension xlsx or csv etc..
df = pd.read_excel(filename+'.xlsx',
                   na_values = ['no info', '.','None','#VALUE!', '#DIV/0!','NA'])
   
df.columns = ['type', 'time', 'Sample', 'tempLED', 'Vbatt', 'plot', 'MZ', 'Nrate', 
              'rep', 'YF_UV', 'RF_UV', 'FRF_UV', 'YF_B','RF_B','FRF_B',
              'YF_G','RF_G', 'FRF_G','YF_R','RF_R','FRF_R']     


## Define Plot no--------------------------------------------------------------
ss = 1
##-----------------------------------------------------------------------------
## Group by Plot numbers
df1 = df[df['plot'] == ss]




##-----------------------------------------------------------------------------
Ym=df1.values

# Read sensor data
YF_UV = np.float64(Ym[:,9]) 
RF_UV = np.float64(Ym[:,10]) 
FRF_UV = np.float64(Ym[:,11]) 
YF_B = np.float64(Ym[:,12]) 
RF_B = np.float64(Ym[:,13]) 
FRF_B = np.float64(Ym[:,14]) 
YF_G = np.float64(Ym[:,15]) 
RF_G = np.float64(Ym[:,16]) 
FRF_G = np.float64(Ym[:,17]) 
YF_R = np.float64(Ym[:,18]) 
RF_R = np.float64(Ym[:,19]) 
FRF_R = np.float64(Ym[:,20]) 


df2 = pd.DataFrame(columns=['YF_UV', 'RF_UV', 'FRF_UV', 'YF_B','RF_B','FRF_B','YF_G','RF_G',
                   'FRF_G','YF_R','RF_R','FRF_R'])






for column in df1[['YF_UV', 'RF_UV', 'FRF_UV', 'YF_B','RF_B','FRF_B','YF_G','RF_G',
                   'FRF_G','YF_R','RF_R','FRF_R']]:
    
    # Select column contents by column  
    # name using [] operator
    columnSeriesObj = df1[column]
    #print('Column Contents : ', columnSeriesObj.values)
    signal = columnSeriesObj.values

    ##-------
    ## Ref: https://github.com/MProx/Wavelet-denoising/blob/master/wavelets.py
    ## Create synthetic signal

    dt = 1.0
    tmax = df1.shape[0]
    t = np.arange(0, tmax, dt)  ##70 HZ--70 cycles per sec.
    minsignal, maxsignal = signal.min(), signal.max()
    ## Compute Fourier Transform
    n = len(signal)


    data = signal
    index = t

    # Create wavelet object and define parameters
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    # maxlev = 2 # Override if desired
    # print("maximum level is " + str(maxlev))
    threshold = 0.5 # Threshold for filtering

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        #plt.subplot(maxlev, 1, i)
        #plt.plot(coeffs[i])
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
        
    datarec = pywt.waverec(coeffs, 'sym4')
    datarec = datarec[0:n]

    # mean_s = np.mean(signal)
    # mean_ds_WT = np.mean(datarec)
    
    this_column = column
    df2[this_column] = datarec
    
    
   
## Export it as an excel file
df2.to_excel(filename+'_Plot'+str(ss)+'_WTdenoised.xlsx')


## Plotting signal vs noise data
cl = 'YF_UV'
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(index, df1[cl], 'b--', label='Raw signal')
plt.plot(index, df2[cl], 'r-', label='De-noised signal') 
plt.xlabel('Sample no.')
plt.ylabel('Fluorescence signal')
plt.legend()
#minsignal, maxsignal = df['YF_UV'].min(), df['YF_UV'].max()
#plt.ylim(minsignal, maxsignal)
plt.tight_layout()
plt.savefig('Denoised_Signal_WT_Plot'+str(ss)+'.png', bbox_inches='tight', dpi=300)
plt.show()

