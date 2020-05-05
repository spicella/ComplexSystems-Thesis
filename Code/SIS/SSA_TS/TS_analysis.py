#!/usr/bin/env python
# coding: utf-8

# In[200]:


import sys
import scipy as sp
import statsmodels.api as sm
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math 
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import scipy.fftpack
import os
from statsmodels.tsa.stattools import adfuller
from numpy import log
from scipy.stats import chisquare
#!{sys.executable} -m pip install pmdarima


#https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/


# In[2]:


main_path  = os.getcwd()
main_path = main_path+"/Matlab/data_sis_v0/SSA/"


# In[3]:


main_path


# In[4]:


#df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv', names=['value'], header=0)
#df.head()


# In[10]:


df = pd.read_csv(main_path+"N100_tend100_beta1.40.csv")
df.columns=['idx','time','value','std']
df = pd.DataFrame(df['value'])
#if P Value > 0.05 we go ahead with finding the order of differencing.

from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(df.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# In[ ]:





# In[89]:


import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})


# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value); axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df.value.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()


# In[57]:


# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3)})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff().diff()); axes[0].set_title('2st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.value.diff().diff().dropna(), ax=axes[1])

plt.show()


# In[137]:


import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff().diff().diff().diff().diff().diff().diff()); axes[0].set_title('7th Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df.value.diff().diff().diff().diff().diff().diff().diff().dropna(), ax=axes[1])

plt.show()


# In[ ]:





# In[ ]:





# In[17]:


for i in range(0,5):
    for j in range(0,3):
        for k in range(0,5):
            try:
                print("(p,d,q)=(%d,%d,%d)"%(i,j,k))
                model = ARIMA(df.value, order=(i,j,k))
                model_fit = model.fit(disp=0)
                print(model_fit.summary())
            except:
                print("nope!")
            print("\n")


# In[12]:


from statsmodels.tsa.arima_model import ARIMA
p,d,q = 1,0,0
model = ARIMA(df.value, order=(p,d,q))
model_fit = model.fit(disp=0)
print(model_fit.summary())


residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()


#Out-of-Time Cross validation
from statsmodels.tsa.stattools import acf
threshold = int(len(df)*.75)
# Create Training and Test
train = df.value[:threshold]
test = df.value[threshold:]

# Build Model
model = ARIMA(train, order=(p, d, q))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

# Forecast
fc, se, conf = fitted.forecast(len(df)-threshold, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5))
plt.plot(train, label='Data - training',alpha = .8,ls='--')
plt.plot(test, label='Data - test',alpha = .8,ls='--')
plt.plot(fc_series, label='Forecast')
plt.hlines((1.-1./5)*500,0,len(df.value),color='r',label="Analytical Solution")

plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15,label="$\\pm95\\%$ c.l.")
plt.title('N=500, $R_{0}=5$')
plt.legend(fontsize=8)
plt.xlabel("time step")
plt.ylabel("I(time step)")
plt.show()


# In[ ]:





# # Following Ivan's 

# In[355]:


main_path  = os.getcwd()
N=100
mu1,mu2 = int(N*.5), int(N*.5)
s1,s2 = int(N*.1), int(N*.1)
datafolder_path = main_path+"/Matlab/data_sis_v2/%d_%d_%d_%d_%d/CME/" %(N,mu1,mu2,s1,s2)


# In[356]:


cme_dfs = []
for filename in os.listdir(datafolder_path):
    if filename.endswith(".csv") and filename[0].isdigit():
        cme_dfs.append([float(filename.split(".csv")[0]),pd.read_csv(datafolder_path+filename,header=None,)])
        
def sortSecond(val): 
    return val[0] 
        
cme_dfs.sort(key =sortSecond )  #sort by increasing c3
r0s = []
for i in range(0,len(cme_dfs)):
    r0s.append(cme_dfs[i][0])
len(r0s)

def chi2(a,b):
    #a = observed, b = expected
    size = len(a)
    res = np.sum(np.divide(np.power(np.subtract(a,b),2),b))
    return res


# In[439]:


main_path  = os.getcwd()
main_path = main_path+"/Matlab/data_sis_v0/SSA/"

#beta!
beta=2.7


ssa_df_2 = pd.read_csv(main_path+"N100_tend100_beta%.2f.csv"%(beta))

ssa_df_2.columns = ['idx','time','value','std']
#ssa_df = pd.DataFrame(np.divide(ssa_df['value'],ssa_df['value'].sum()))

ssa_df_2 = pd.DataFrame(np.add(-ssa_df_2['value'],N))

result = adfuller(ssa_df_2.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# In[440]:


for i in range(0,len(r0s)):
    data = cme_dfs[i][1].sum(axis=0)/sum(cme_dfs[i][1].sum(axis=0))
    plt.plot(data)


# In[ ]:





# In[441]:


#Obtain 1D dfs for the PDFs
data_pdf = []
for i in range(0,len(r0s)):
    data = cme_dfs[i]
    data = data[1].sum(axis=0)/sum(data[1].sum(axis=0))
    
    pdf_df = pd.DataFrame([np.arange(0,100,1),data]).T
    pdf_df.columns=['S','value']
    data_pdf.append(pdf_df)


# In[ ]:





# In[442]:


# Obtain 1D pdf for SSA
counts, bins, bars = plt.hist(ssa_df_2['value'],bins=int(max(ssa_df_2['value'])-min(ssa_df_2['value'])));
plt.show()


# In[443]:


#Convert SSA to appropriate df
counts = np.divide(counts,counts.sum())
plt.plot(bins[1:],counts,label="SSA, $R_{0}=%.2f$"%(beta))
plt.legend()

t_true = np.arange(0,100,1)
data_ssa = np.zeros(len(data2))

for i in range(0,len(counts)):
    data_ssa[i+int(min(bins))]=counts[int(i)]
df_ssa = pd.DataFrame([t_true,data_ssa]).T
df_ssa.columns=['S','value']


# In[444]:


#idx = 50
#plt.scatter(xrange[idx-min(xrange)],counts[0],marker='s',s=50,label="hist")
#plt.scatter(xrange[idx-min(xrange)],counts[idx-min(xrange)],marker='s',s=50,label="hist")

#plt.scatter(idx,data2[idx],marker='X',s=100,label="pdf")
#plt.plot(bins[1:],counts,label='hist',ls='--',alpha=.4)
#for i in xrange:
#    plt.scatter(i,data2[i],s=5)
#plt.legend()


# In[445]:


main_path  = os.getcwd()
output_folder = main_path+"/Matlab/data_sis_v0/SSA"

chi2_array = []
for i in range(0,len(r0s)):
    
    #plt.plot(pdf_df['S'],data_pdf[i]['value'],label='CME')
    #plt.plot(df_ssa['S'],df_ssa['value'],label='SSA')
    chi2_array.append(chi2(df_ssa['value'],data_pdf[i]['value']))
    #print("R0 = %.2f => chi2 = %.2e"%(r0s[i],chi2_array[-1]))
    #plt.legend()
    #plt.show()
plt.figure(figsize=[16,7])
plt.scatter(r0s[np.argmin(chi2_array)],min(chi2_array),color='red',label="min($\\chi^{2}$)",s=80)
plt.plot(r0s,chi2_array,ls='--',marker='.',label="$\\chi^{2}(R_{0})$")
plt.vlines(beta,0,max(chi2_array),label='Actual $R_{0}$ of the SSA',ls='-',color='b')
plt.semilogy()
plt.legend()
plt.title("$\\chi^{2}$ minimization: CME - SSA comparison\n N = %d, $R_{0,SSA} = %.2f$"%(N,beta),fontsize=20)
plt.xlabel("$R_{0}$",fontsize=16)
plt.ylabel("$\\chi^{2}$",fontsize=16)
plt.grid()
plt.savefig(output_folder+"/%d_%.2f_mle.png" %(N,beta))
plt.show()


# In[446]:


plt.figure(figsize=[16,7])
plt.title("CME - SSA comparison N = %d, \n$R_{0,SSA} = %.2f$, best fit $R_{0,CME} = %.2f$" %(N,beta,r0s[np.argmin(chi2_array)]),fontsize=20)
plt.xlabel("$S$",fontsize=16)
plt.ylabel("Normalized distributions",fontsize=16)
plt.grid()
plt.plot(data_pdf[np.argmin(chi2_array)]['S'],data_pdf[np.argmin(chi2_array)]['value'],label='CME')
plt.plot(df_ssa['S'],df_ssa['value'],label='SSA')
plt.legend(fontsize=16)
plt.savefig(output_folder+"/%d_%.2f_distr.png" %(N,beta))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




