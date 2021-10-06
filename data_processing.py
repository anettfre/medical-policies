import pandas as pd
import numpy as np
#import seaborn as sn
import matplotlib.pyplot as plt

features = pd.read_csv('../../data/medical/historical_X.dat') 
actions = pd.read_csv('../../data/medical/historical_A.dat')
results =  pd.read_csv('../../data/medical/historical_Y.dat')

conf_mat = pd.crosstab(actions['1'], results['1'], rownames=['Action'], colnames=['Result'])
print(conf_mat)
#ax = sn.heatmap(conf_mat)
#plt.show()
cured = 69/(7633+69)
print(conf_mat[0][1])
print("Percentage of patients cured with trick: %f" %(conf_mat[1][0]/(conf_mat[0][0]+conf_mat[1][0])*100))
print("Percentage of patients cured with treatment: %f" %(conf_mat[1][1]/(conf_mat[1][1]+conf_mat[0][1])*100))
"""
We assume that result=0 means that the patient is not cured and that result=1 means that the patient is cured.
First we make a confusion matrix to see whether our assumptions are right.

      Result 0     1
Action
0          7633    69
1           946  1351

Looking at the matrix, it seesm likely that the ones put on placebo has a much larger discrepancy between who's treated and who gets well than the ones given a treatment.
Further, we can look at the percentage of patients cured by trick vs treatment
Percentage of patients cured with trick: 0.89%
Percentage of patients cured with treatment: 58.81%
"""

