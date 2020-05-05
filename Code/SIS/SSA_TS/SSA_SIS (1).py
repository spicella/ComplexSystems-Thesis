#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math 
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import scipy.fftpack
import os


# In[2]:


main_path  = os.getcwd()


# In[5]:


def find_reaction(a,asum,random):
    summ = 0
    found = False
    index = [0] #first reaction
    while found==False:
        for i in index:
            try:
                summ += a[i] #compute sum from first index up
            except: #if number of possible reactions exceeded, take last
                found=True 
                return index[-2]
        if summ>random*asum: #condition met
            found=True
            return index[-1]
        else:
            #add the following new index to the summation and try again
            index.append(index[-1]+1)

def deriv_sis(y, t, beta, gamma):
    S, I = y
    dSdt = gamma * I - beta * S * I 
    dIdt = beta * S * I - gamma * I
    return dSdt, dIdt

def func(x, a, c, d):
    return a*np.exp(-c*x)+d


# In[4]:


### Initial conditions
X = np.zeros(2) #two species
N = 100
vol = N

X0_0 = .1*N
X1_0 =  N-X0_0
X[0] = X0_0
X[1] = X1_0
a = np.zeros(2) #two reactions
V = np.array([[-1, 1], [1, -1]]) #stechiometric matrix
beta,gamma= 2,1

output_folder = main_path+"/Matlab/data_sis_v0/SSA"

### Start simulation SSA
history = [] #[t, X[0],X[1]]
t = 0
tau=0
t_end = 500
jlist = []
while t<t_end:
    history.append([t, X[0],X[1],tau]) # t, S, I, tau
    #evaluate propensities
    a[0] = beta * X[0] * X[1] / vol   #(X[0],X[1]) -> (X[0]-1,X[1]+1)
    a[1] = gamma * X[1]         #(X[0],X[1]) -> (X[0]+1,X[1]-1)
    asum = np.sum(a)

    #evaluate reaction and tau
    rand = np.random.uniform(size=2)
    tau = math.log(1/rand[1])/asum
    #naive
    #if rand[0] < a[0] / asum:
    #    j=0
    #else:
    #    j=1
    j=find_reaction(a,asum,rand[0])

    jlist.append(j)
    #Update
    X += V[:,j] 
    t = t+tau
    #Exit condition: # of one of the species goes to zero
    if np.prod(X)==0:
        t=t_end

### SaveData SSA & fit parameters
history = pd.DataFrame(history)
popt, pcov = curve_fit(func, history[0], history[2], p0=(N, 1, 1/beta))
tfit = np.linspace(0,history[0].iloc[-1], 1000)
fitted =  func(tfit, *popt)
print(popt)

history.to_csv(output_folder+"/%d_%d_%d_%.2e.csv" %(N,X0_0,X1_0,beta/gamma))


### Start ODE 
S0, I0 = X0_0/N, X1_0/N
y0 = S0, I0
t = np.linspace(0, t_end, 1000)

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv_sis, y0, t, args=(beta, gamma))
S, I = ret.T
if beta>gamma:
    print("Equilibrium @ %.2f"%(1-gamma/beta))
else:
    print("Exponential decay")


# In[108]:


### Plot
plt.figure(figsize=(12,8))
plt.title("SSA on SIS model, $(S_{0},I_{0})=(%d,%d)$, $R_{0}=%.2f$, $t_{tot}=%d$"%(X0_0,X1_0,beta/gamma,history[0].iloc[-1]),fontsize=16)
plt.plot(history[0],history[2],label="I(t), SSA",alpha = .3,color='green')
plt.plot(t,I*N,label="I(t), ODE",ls='--',color='red',alpha=.7,linewidth=3)
mov_avg = 10000
rolling_mean = history[2].rolling(window=mov_avg).mean()
plt.plot(history[0],rolling_mean, ls='--',color='blue', label="I(t), SSA, moving average @%d"%(mov_avg))
plt.legend(fontsize=12)
plt.xlabel("t",fontsize=14)
plt.ylabel("Number of species",fontsize=14)
plt.grid()
plt.ylim(0,N)

### SavePlot
plt.savefig(output_folder+"/%d_%d_%d_%.2e.png" %(N,X0_0,X1_0,beta/gamma))
plt.show()

plt.plot(tfit,fitted,label="Fit")
plt.plot(t,I*N,label="I(t), ODE",ls='--',color='red',alpha=.7,linewidth=3)
plt.legend()
plt.ylim(0,N)
plt.grid()


# In[ ]:





# In[66]:


output_folder = main_path+"/Matlab/data_sis_v0/SSA"

def loop_ssa(nruns,beta,S0_frac=.1,N=100,t_end=500):
    popts = []
    pcovs = []
    for i in range(0,nruns):
        ### Initial conditions
        X = np.zeros(2) #two species
        N = 100
        vol = N

        X0_0 = S0_frac*N
        X1_0 =  N-X0_0
        X[0] = X0_0
        X[1] = X1_0
        a = np.zeros(2) #two reactions
        V = np.array([[-1, 1], [1, -1]]) #stechiometric matrix
        gamma= 1

        output_folder = main_path+"/Matlab/data_sis_v0/SSA"

        ### Start simulation SSA
        history = [] #[t, X[0],X[1]]
        t = 0
        tau=0
        t_end = 200
        jlist = []
        while t<t_end:
            history.append([t, X[0],X[1],tau]) # t, S, I, tau
            #evaluate propensities
            a[0] = beta * X[0] * X[1] / vol   #(X[0],X[1]) -> (X[0]-1,X[1]+1)
            a[1] = gamma * X[1]         #(X[0],X[1]) -> (X[0]+1,X[1]-1)
            asum = np.sum(a)

            #evaluate reaction and tau
            rand = np.random.uniform(size=2)
            tau = math.log(1/rand[1])/asum
            #naive
            #if rand[0] < a[0] / asum:
            #    j=0
            #else:
            #    j=1
            j=find_reaction(a,asum,rand[0])

            jlist.append(j)
            #Update
            X += V[:,j] 
            t = t+tau
            #Exit condition: # of one of the species goes to zero
            if np.prod(X)==0:
                t=t_end

        ### SaveData SSA & fit parameters
        history = pd.DataFrame(history)
        popt, pcov = curve_fit(func, history[0], history[2], p0=(N, 1, 1/beta))
        tfit = np.linspace(0,history[0].iloc[-1], 1000)
        fitted =  func(tfit, *popt)
        popts.append(popt)
        pcovs.append(pcov)

        history.to_csv(output_folder+"/%d_%d_%d_%.2e.csv" %(N,X0_0,X1_0,beta/gamma))


        print("Finished %d of %d"%(i+1,nruns))
    S0, I0 = X0_0/N, X1_0/N
    y0 = S0, I0
    t = np.linspace(0, t_end, 1000)

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_sis, y0, t, args=(beta, gamma))
    S, I = ret.T
    return popts, pcovs, S, I, t_end, N, S0_frac


# # SSA @ stabilization (nofit)

# In[6]:


output_folder = main_path+"/Matlab/data_sis_v0/SSA"

def loop_ssa_equil(nruns,beta,S0_frac=.1,N=100,t_end=1000,lasts=10000):
    histories = [] #collection of multiple runs
    for i in range(0,nruns):
        ### Initial conditions
        X = np.zeros(2) #two species
        vol = N

        X0_0 = S0_frac*N
        X1_0 =  N-X0_0
        X[0] = X0_0
        X[1] = X1_0
        a = np.zeros(2) #two reactions
        V = np.array([[-1, 1], [1, -1]]) #stechiometric matrix
        gamma= 1

        output_folder = main_path+"/Matlab/data_sis_v0/SSA"

        ### Start simulation SSA
        history = [] #[t, X[0],X[1]]
        t = 0
        tau=0
        jlist = []
        while t<t_end:
            #if t>.9*t_end:
            history.append([t, X[0],X[1],tau]) # t, S, I, tau
            #evaluate propensities
            a[0] = beta * X[0] * X[1] / vol   #(X[0],X[1]) -> (X[0]-1,X[1]+1)
            a[1] = gamma * X[1]         #(X[0],X[1]) -> (X[0]+1,X[1]-1)
            asum = np.sum(a)

            #evaluate reaction and tau
            rand = np.random.uniform(size=2)
            tau = math.log(1/rand[1])/asum
            #naive
            #if rand[0] < a[0] / asum:
            #    j=0
            #else:
            #    j=1
            j=find_reaction(a,asum,rand[0])

            jlist.append(j)
            #Update
            X += V[:,j] 
            t = t+tau
            #Exit condition: # of one of the species goes to zero
            if np.prod(X)==0:
                t=t_end

        ### SaveData SSA & fit parameters
        len_tot_hist = len(history)
        history = pd.DataFrame(history[-lasts:])


        #history.to_csv(output_folder+"/%d_%d_%d_%.2e.csv" %(N,X0_0,X1_0,beta/gamma))
        histories.append(history)
        print("Finished %d of %d of length %d"%(i+1,nruns,len_tot_hist))
        plt.plot(history[0],history[2],alpha=.1)
    if beta>1:
        plt.hlines((1.-1./beta)*N,history[0].iloc[0],history[0].iloc[-1],color='r',label="Analytical Solution")
    plt.legend()
    plt.show()
    return histories, t_end, N, S0_frac,lasts, beta


# In[10]:


N_sim = 100
nruns_sim = 1
t_end_sim = 100

lasts_sim = 10000

for i in np.arange(0,3.1,.1):
    res = loop_ssa_equil(nruns=nruns_sim, beta=i,N=N_sim, t_end=t_end_sim, lasts=lasts_sim)
    pdf_df = []
    for i in range(0, nruns_sim):
        pdf_df.append(res[0][i][2])
    pdf_df = pd.DataFrame(pdf_df)
    pdf_mean = []
    pdf_std = []
    for i in range (0,np.shape(pdf_df)[1]):
        pdf_mean.append(pdf_df[i].mean())
        pdf_std.append(pdf_df[i].std())
    t = np.arange(0,len(pdf_mean),1)
    plt.figure(figsize=[12,8])
    plt.title("Mean$\\pm \sigma$ (last %d steps of each run)"%(res[4]))
    plt.errorbar(t,pdf_mean,yerr=np.multiply(pdf_std,2),alpha=.7,linestyle='',capsize=.8,elinewidth=.1,marker='',label='$\\pm 2\\sigma$')
    plt.errorbar(t,pdf_mean,yerr=pdf_std,capsize=.8,alpha=.8,elinewidth=.1,linestyle='',marker='',label='$\\pm\\sigma$')
    plt.plot(t,pdf_mean,label='Avg')
    if i>1:
        plt.hlines((1.-1./res[5])*res[2],0,len(pdf_mean),color='r',linestyle='--',label="Analytical Solution")

    plt.title("$R_{0}$ = %.3f$, $N = %d"%(res[5], res[2]))


    plt.legend()
    plt.xlabel("t")
    plt.ylabel("I(t)")
    plt.show()

    pdf_df = pd.DataFrame([t,pdf_mean,pdf_std])
    (pdf_df.T).to_csv(output_folder+"/N%d_tend%d_beta%.2f.csv"%(res[2],res[1],res[5]))

    plt.xlabel("I(t)")
    plt.ylabel("Frequency")
    plt.hist(pdf_mean,bins=20,alpha=.8)
    plt.grid()
    plt.show()


# In[32]:





# ### Using GMM

# In[9]:


#Create Matrix of data

data_mat = pd.DataFrame(res[0][0][2])
for i in range(0,nruns_sim):
    data_mat[i] =  pd.DataFrame(res[0][i][2])

#Define model
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
import scipy.stats as stats



# In[70]:


#Reshape PDF
pdf_data = np.asarray(pdf)
pdf_data = np.reshape(pdf_data,[len(pdf_data),1])

#Model with Gaussian Mixture
gmm = GaussianMixture(n_components=1, covariance_type ='full',                   init_params = 'random', max_iter = 100, random_state=0)
gmm.fit(pdf)

weights = gmm.weights_
means = gmm.means_
covars = gmm.covariances_

print("Means:")
print(means)
print("Weights:")
print(weights)
print("Covs:")
print(covars)

#Plot
D = pdf_data.ravel()
xmin = D.min()
xmax = D.max()
x = np.linspace(xmin,xmax,1000)

for i in range(0,len(means)):

    mean = means[i]
    sigma = math.sqrt(covars[i])
    plt.plot(x,weights[i]*stats.norm.pdf(x,mean,sigma))


#plt.savefig("DataGMM.png")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




