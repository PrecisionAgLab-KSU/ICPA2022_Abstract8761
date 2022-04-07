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


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.pipeline import make_pipeline
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
plt.rcParams['figure.figsize'] = [6,4]
plt.rcParams.update({'font.size': 16})

########################

########################

Y1 = pd.read_excel('Ardec2012_Regression_Data.xlsx', sheet_name='trainv9', 
                   na_values = ['no info', '.','None','#VALUE!', '#DIV/0!'],
                   skiprows=[0],header=None)
Ym=Y1.values
Ym0=np.float64(Ym[:,12])#N%
Ym1=np.float64(Ym[:,13])#Nuptake
Ym2=np.float64(Ym[:,14])#Biomass
#Y=np.column_stack((Ym0,Ym2))
Y=Ym2

## Predictors--WT denoised
NBI_R = np.float64(Ym[:,3])
NBI_G = np.float64(Ym[:,4])
NBI_B = np.float64(Ym[:,5])
NBI1 = np.float64(Ym[:,6])
CHL = np.float64(Ym[:,7])
CHL1 = np.float64(Ym[:,8])
FLAV = np.float64(Ym[:,9])


X=np.column_stack((NBI_R, NBI_G, NBI_B, NBI1, CHL, CHL1, FLAV))


##SVR model training-----------------------------------------------------------
pipeline = make_pipeline(StandardScaler(),
    SVR(kernel='rbf', epsilon=1.5, C=200, gamma = 0.5),
)
MSVRmodel=pipeline.fit(X,Y)

## Predicting on Training data-------------------------------------------------
y_out1 = pipeline.predict(X);

######################################3
##PAI estimation and error
#rmse estimation
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse_value1 = rmse(np.array(Ym2), np.array(y_out1))
#Correlation coefficient 
corrr_value=np.corrcoef(np.array(Ym2), np.array(y_out1))
rr_value1= corrr_value[0,1]
mae_value1 = mean_absolute_error(Ym2,y_out1)


##-----------------------------------------------------------------------------
## Model cross validation score------------------------------------------------
model = pipeline
# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

## https://scikit-learn.org/stable/modules/model_evaluation.html
n_scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

n_scores = cross_val_score(model, X, Y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('RMSE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

n_scores1 = cross_val_score(model, X, Y, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')
n_scores = np.sqrt(np.abs(n_scores1))
# report performance
print('R: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))





###############################################################################
###Load validation data--------------------------------------------------------
cornval = pd.read_excel('Ardec2012_Regression_Data.xlsx', sheet_name='testv9', 
                        na_values = ['no info', '.','None','#VALUE!', '#DIV/0!'],
                        skiprows=[0],header=None)

vald=cornval.dropna(subset=[1])                                                                                                                            
valdm=vald.values

valN=np.float64(valdm[:,12])#N%
valNup=np.float64(valdm[:,13])#Nuptake
valbiom=np.float64(valdm[:,14])#Biomass
valY = valbiom

## Predictors--WT denoised
vNBI_R = np.float64(valdm[:,3])
vNBI_G = np.float64(valdm[:,4])
vNBI_B = np.float64(valdm[:,5])
vNBI1 = np.float64(valdm[:,6])
vCHL = np.float64(valdm[:,7])
vCHL1 = np.float64(valdm[:,8])
vFLAV = np.float64(valdm[:,9])

valX=np.column_stack((vNBI_R, vNBI_G, vNBI_B, vNBI1, vCHL, vCHL1, vFLAV))


# Prediction for validation data
y_out = pipeline.predict(valX);

##-----------------------------------------------------------------------------
##estimation and error
#rmse estimation
rmse_value2 = rmse(np.array(valbiom), np.array(y_out))
#Correlation coefficient 
corrr_value=np.corrcoef(np.array(valbiom), np.array(y_out))
rr_value2= corrr_value[0,1]
mae_value2 = mean_absolute_error(valbiom,y_out)



#------------------------------------------------------------------------------
#Plotting estimates over train and test data
matplotlib.rcParams.update({'font.size': 14})
plt.plot(Ym2,y_out1, 'ro', markersize=5, markerfacecolor='None', markeredgecolor='r', label = "Train")
plt.plot(valbiom,y_out, 'gd', label = "Test")
plt.xlabel("Observed biomass ($g$)")
plt.ylabel("Estimated biomass ($g$)")
y_ticks = np.arange(0, 200, 50)
plt.yticks(y_ticks)
x_ticks = np.arange(0, 200, 50)
plt.xticks(x_ticks)
plt.plot([0, 150], [0, 150], 'k:', label = "1:1 line")
plt.gca().set_aspect('equal', adjustable='box')
matplotlib.rcParams.update({'font.size': 13})
plt.annotate('r = %.2f (Train)/'%rr_value1, xy=(0.05, 145))#round off upto 2decimals
plt.annotate('%.2f (Test)'%rr_value2, xy=(80, 145))#round off upto 2decimals
plt.annotate('RMSE = %.2f (Train)/'%rmse_value1, xy=(0.05, 135))
plt.annotate('%.2f (Test)'%rmse_value2, xy=(43, 125))
plt.annotate('MAE = %.2f (Train)/'%mae_value1, xy=(0.05, 115))
plt.annotate('%.2f (Test)'%mae_value2, xy=(38, 105))
plt.legend(loc='lower right')
plt.savefig('Regression_Biomass_V9.png',bbox_inches="tight",dpi=450)
plt.show()


