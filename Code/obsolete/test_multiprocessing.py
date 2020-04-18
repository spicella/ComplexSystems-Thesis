#!/usr/bin/env python
# coding: utf-8

# In[242]:


import numpy as np
import pandas as pd
import math
import time
import timeit
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def time_stuff(fn):
    """
    Measure time of execution of a function
    """
    def wrapper(*args, **kwargs):
        t0 = timeit.default_timer()
        fn(*args, **kwargs)
        t1 = timeit.default_timer()
        print("{} seconds".format(t1 - t0))
    return wrapper


# # Shifting matrix==> no better result using multithread

# In[226]:


def find_primes_in(nmin, nmax):
    """
    Compute a list of prime numbers between the given minimum and maximum arguments
    """
    primes = []

    # Loop from minimum to maximum
    for current in range(nmin, nmax + 1):

        # Take the square root of the current number
        sqrt_n = int(math.sqrt(current))
        found = False

        # Check if the any number from 2 to the square root + 1 divides the current numnber under consideration
        for number in range(2, sqrt_n + 1):

            # If divisible we have found a factor, hence this is not a prime number, lets move to the next one
            if current % number == 0:
                found = True
                break

        # If not divisible, add this number to the list of primes that we have found so far
        if not found:
            primes.append(current)
    return primes
    # I am merely printing the length of the array containing all the primes, but feel free to do what you want
    #do whatever here

def process_executor_prime_finder(nmin, nmax):
    """
    Split the min max interval similar to the threading method, but use the process pool executor.
    This is the fastest method recorded so far as it manages process efficiently + overcomes GIL limitations.
    RECOMMENDED METHOD FOR CPU-BOUND TASKS
    """
    nrange = nmax - nmin
    max_workers= 8
    with ProcessPoolExecutor(max_workers = max_workers) as e:
        out_result = []
        for i in range(max_workers):
            start = int(nmin + i * nrange/8)
            end = int(nmin + (i + 1) * nrange/8)
            out_result.append((e.submit(find_primes_in, start, end).result()))
    return out_result


# In[307]:


def convert_to_df(mat,dtype="float32"): 
    #dataframes were used as they allowed computational times almost
    #as fast as numpy but with easier syntax/more useful built in functions 
    #for the present algorithm
    return pd.DataFrame(np.float32(mat))

def convert_to_df(mat,dtype="float32"): 
    #dataframes were used as they allowed computational times almost
    #as fast as numpy but with easier syntax/more useful built in functions 
    #for the present algorithm
    return pd.DataFrame(np.float32(mat))

@time_stuff
def get_distrib_mat(grid):
    # This is the only true parallelizable process
    """Here only one step is needed in a given direction"""
    """but this can be easily be changed and n-steps in whatever direction"""
    """Given a grid (of PDF), returns values of the grid (n) steps in a given"""
    """direction of the grid."""
    df_grid = convert_to_df(grid)
    N = df_grid.shift(-1,axis=0,fill_value=0)  #looks 1 cell up
    E = df_grid.shift(-1,axis=1,fill_value=0) #looks 1 cell right
    S = df_grid.shift(1,axis=0,fill_value=0) #looks 1 cell down 
    W = df_grid.shift(1,axis=1,fill_value=0)  #looks 1 cell left 
    return N,E,S,W

def get_distrib_mat_multithread(grid,ax,direction):
    # This is the only true parallelizable process
    """Here only one step is needed in a given direction"""
    """but this can be easily be changed and n-steps in whatever direction"""
    """Given a grid (of PDF), returns values of the grid (n) steps in a given"""
    """direction of the grid."""
    df_grid = convert_to_df(grid)
    return df_grid.shift(direction,axis=ax,fill_value=0) 


# In[308]:


@time_stuff
def shift_grid_multithread(grid):
    """
    Split the min max interval similar to the threading method, but use the process pool executor.
    This is the fastest method recorded so far as it manages process efficiently + overcomes GIL limitations.
    RECOMMENDED METHOD FOR CPU-BOUND TASKS
    """
    df_grid = convert_to_df(grid)
    max_workers= 2
    counter = 0
    c=0
    with ProcessPoolExecutor(max_workers = max_workers) as e:
        out_result = []
        for i in range(max_workers):
            if c==0:
                ax = 0 #check NS direction
                pm = -1 #look one up
            elif c==1:
                ax = 1 #check WE direction
                pm = -1 #look one east
            elif c==2:
                ax = 0 #check NS direction
                pm = 1 #look one down
            elif c==3:
                ax = 1 #check WE direction
                pm = 1 #look one west    
            c+=1

            out_result.append((e.submit(get_distrib_mat_multithread,grid,ax, pm).result()))
    return out_result


# In[321]:


size = [300,300]
mat_test = np.random.random(size=size)


# In[322]:


get_distrib_mat(mat_test) #original (pandas)
shift_grid_multithread(mat_test) #multithread


# #### In conclusion, not efficient to multithread the shift operations

# ### Trying with 4 matrix multiplications in parallel

# In[552]:


from threading import Thread
import queue
from time import sleep


# In[620]:


size=300

a = np.float32(np.random.random(size=[size,size]))
b = np.float32(np.random.random(size=[size,size]))
c = np.float32(np.random.random(size=[size,size]))


def increment_direction(a,b,c,d): #q=queue value
    return np.multiply(a,b)- np.multiply(c,d)


def wrapper(func, args, res):
    res.append(func(*args))
def eval_increments(a,b,c,d):
    res = []
    t1 = threading.Thread(
        target=wrapper, args=(increment_direction, (a,b,c,c), res))
    t2 = threading.Thread(
        target=wrapper, args=(increment_direction, (a,c,b,b), res))
    t3 = threading.Thread(
        target=wrapper, args=(increment_direction, (a,c,a,c), res))
    t4 = threading.Thread(
        target=wrapper, args=(increment_direction, (a,c,a,a), res))
    t1.start()
    t2.start()    
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    
    #return res for the individual matrices
    return np.add(np.add(res[0],res[1]),np.add(res[0],res[1]))


# In[629]:


#"k=" just to simulate the declaration of variable 
n=1
start = time.clock()
for i in range(0,n):
    avg = 0
    k = eval_increments(a,b,a,c)
    avg += time.clock()-start
avg/=n
print(avg)

start = time.clock()
for i in range(0,n):
    avg = 0
    k1 = np.multiply(a,b)-np.multiply(a,b)
    k2 = np.multiply(a,c)-np.multiply(a,b)
    k3 = np.multiply(c,b)-np.multiply(a,b)
    k4 = np.multiply(a,a)-np.multiply(a,b)
    
    result = np.add(k1,k2),np.add(k3,k4)
    avg += time.clock()-start
avg/=n
print(avg)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def increment(grid,a0,a1,a2,a3,a4,a0w,a1e,a2n,a3s):
    """
    Split the min max interval similar to the threading method, but use the process pool executor.
    This is the fastest method recorded so far as it manages process efficiently + overcomes GIL limitations.
    RECOMMENDED METHOD FOR CPU-BOUND TASKS
    """
    df_grid = convert_to_df(grid)
    max_workers= 4
    counter = 0
    c=0
    with ProcessPoolExecutor(max_workers = max_workers) as e:
        out_result = []
        for i in range(max_workers):
            if c==0:
                ax = 0 #check NS direction
                pm = -1 #look one up
            elif c==1:
                ax = 1 #check WE direction
                pm = -1 #look one east
            elif c==2:
                ax = 0 #check NS direction
                pm = 1 #look one down
            elif c==3:
                ax = 1 #check WE direction
                pm = 1 #look one west    
            c+=1

            out_result.append((e.submit(get_distrib_mat_multithread,grid,ax, pm).result()))
    return out_result


# # Matrix multiplication benchmarks 

# In[323]:


def convert_to_df(mat,dtype="float32"): 
    #dataframes were used as they allowed computational times almost
    #as fast as numpy but with easier syntax/more useful built in functions 
    #for the present algorithm
    return pd.DataFrame(np.float32(mat))

def get_distrib_mat(grid):
    # This is the only true parallelizable process
    """Here only one step is needed in a given direction"""
    """but this can be easily be changed and n-steps in whatever direction"""
    """Given a grid (of PDF), returns values of the grid (n) steps in a given"""
    """direction of the grid."""
    df_grid = convert_to_df(grid)
    N = df_grid.shift(-1,axis=0,fill_value=0)  #looks 1 cell up
    E = df_grid.shift(-1,axis=1,fill_value=0) #looks 1 cell right
    S = df_grid.shift(1,axis=0,fill_value=0) #looks 1 cell down 
    W = df_grid.shift(1,axis=1,fill_value=0)  #looks 1 cell left 
    return N,E,S,W


# In[504]:


avg = 0
n=1000
for i in range(0,n):
    start = time.clock()
    shifted = get_distrib_mat(mat_test)
    N = shifted[0]
    E = shifted[1]
    S = shifted[2]
    W = shifted[3]
    avg += time.clock()-start
avg/=n
print(avg)

    #obviously, first approach faster than second of about a factor 2-4
avg = 0
for i in range(0,n):
    start = time.clock()
    N = get_distrib_mat(mat_test)[0]
    E = get_distrib_mat(mat_test)[1]
    S = get_distrib_mat(mat_test)[2]
    W = get_distrib_mat(mat_test)[3]
    avg += time.clock()-start
avg/=n
print(avg)


# In[393]:


avg = 0
n=1000
for i in range(0,n):
    #also here, a factor 5 when using np
    
    start = time.clock()
    a = np.multiply(N,N)
    avg += time.clock()-start
avg/=n
print(avg)

avg = 0
for i in range(0,n):
    start = time.clock()
    a = N*N
    avg += time.clock()-start
avg/=n
print(avg)


# #### Benchmark for array2d*pandas multiplication

# In[399]:


size = [300,300]
mat_test = np.random.random(size=size)
shifted = get_distrib_mat(mat_test)
N = shifted[0] #DataFrame type
E = shifted[1]
S = shifted[2]
W = shifted[3]

avg = 0
n=1000
for i in range(0,n):
    start = time.clock()
    N*mat_test
    avg += time.clock()-start
avg/=n
print(avg)

N=np.float32(N) #significant (20x) improve if np*np multiplication instead of pd*np is used
avg = 0
n=1000
for i in range(0,n):
    start = time.clock()
    N*mat_test
    avg += time.clock()-start
avg/=n
print(avg)


# In[380]:





# In[ ]:


#differences in the order of 4/1000, method already implemented is faster..


# In[ ]:





# In[ ]:





# In[437]:


c1=c4 = 3e3
c2=c5 = 1.1e4
c3=c6 = 1e-3
beta = gamma = 2



def r1(i,j):
    return c1/(c2+j**beta)
def r2(i,j):
    return c3*i
def r3(i,j):
    return c4/(c5+i**gamma)
def r4(i,j):
    return c6*j

size=300
mat_test = np.random.random(size=[size,size])


a = np.zeros(shape=[4,size,size],dtype="float32")
for i in range(0,size):
    for j in range(0,size):
        a[0,i,j] = r1(i,j)
        a[1,i,j] = r2(i,j)
        a[2,i,j] = r3(i,j) 
        a[3,i,j] = r4(i,j) 


# In[438]:


adf = pd.DataFrame(a[0])


# In[442]:


avg = 0
n=50000

for i in range(0,n):
    start = time.clock()
    adf*mat_test
    avg += time.clock()-start
avg/=n
print(avg)


for i in range(0,n):
    start = time.clock()
    a[0]*mat_test
    avg += time.clock()-start
avg/=n
print(avg)

avg = 0
for i in range(0,n):
    start = time.clock()
    np.multiply(a[0],mat_test)
    avg += time.clock()-start
avg/=n
print(avg)
#almost a factor 20!


# In[ ]:




