#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd #easier to operate with, numpy based..!
from ipywidgets import interact
import timeit
import time
import os


# In[2]:


#Folder and paths definitions
main_path  = os.getcwd()
datafolder_path = main_path+"/results"
results_dir = "/output_py" 
output_dir = main_path+results_dir
try:
    os.mkdir(output_dir)
except OSError:
    print ("Creation of the directory %s failed" % results_dir)
else:
    print ("Successfully created the directory %s " % results_dir)


# In[ ]:


def convert_to_df(mat): 
    #dataframes were used as they allowed computational times almost
    #as fast as numpy but with easier syntax/more useful built in functions 
    #for the present algorithm
    return pd.DataFrame(mat)

def makeGaussian(size = 100, fwhm = 50, center=None):
    """ Make a square gaussian kernel 2d."""
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def get_distrib_mat(grid):
    """Here only one step is needed in a given direction"""
    """but this can be easily be changed and n-steps in whatever direction"""
    """Given a grid (of PDF), returns values of the grid (n) steps in a given"""
    """direction of the grid."""
    df_grid = convert_to_df(grid)
    N = df_grid.shift(1,axis=0,fill_value=0)  #looks 1 cell up
    E = df_grid.shift(-1,axis=1,fill_value=0) #looks 1 cell right
    S = df_grid.shift(-1,axis=0,fill_value=0) #looks 1 cell down 
    W = df_grid.shift(1,axis=1,fill_value=0)  #looks 1 cell left 
    return N,E,S,W


#Coefficients propensity functions, from
#"Solution of the chemical master equation by
#radial basis functions approximation with
#interface tracking", Kryven et al

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

#mapping reaction functions
#Note that these coefficients are fixed during the exp, can be evaluated even just once!

#all weights for the reactions, here just examples to picture dynamics of time evolution
size = 100
a = np.zeros(shape=[4,size,size])

for i in range(0,size):
    for j in range(0,size):
        a[0,i,j] = r1(i,j)
        a[1,i,j] = r2(i,j)
        a[2,i,j] = r3(i,j) 
        a[3,i,j] = r4(i,j) 

a_df=[]
a_df.append(convert_to_df(a[0,:,:]))
a_df.append(convert_to_df(a[1,:,:]))
a_df.append(convert_to_df(a[2,:,:]))
a_df.append(convert_to_df(a[3,:,:]))


# In[4]:


plt.matshow(a_df[0]+a_df[1]+a_df[2]+a_df[3])
plt.title("Reaction weights",y=1.1)
plt.colorbar()
plt.show()


# In[5]:


#Coefficients matrices
for i in range(0,4):
    plt.matshow(a_df[i])
    plt.title("Reaction %d weights"%(i+1),y=1.1)
    plt.colorbar()
    plt.show()
    #plt.clim(0, .5);


# In[146]:


def create_size_folderdata(size):
    try:
        os.mkdir(output_dir+"/size=%d"%(size))
    except OSError:
        print ("")
    else:
        print ("Successfully created the directory %s " % output_dir+"/size=%d"%(size))
    return output_dir+"/size=%d"%(size)

#create_name/get data functions

def name_data(t_end):
    """Create name of output file given simulation parameters"""
    data_name = "data_size%d_x0y0=%d_%d_std=%d_t_end=%d.dat"%(size,center[0],center[1],std,t_end)
    return data_name

def get_data(output_path,t_end):
    """Get output file given simulation parameters"""
    data_name = "data_size%d_x0y0=%d_%d_std=%d_t=%d.dat"%(size,center[0],center[1],std,t_end)
    data = pd.read_csv(output_path,header=None,sep=",")
    return data


#visualization functions


def display_sequence_contourf(output_path,t_end):
    """Slider implementation of visualization over time"""
    data = get_data(output_path,t_end)
    def _show(frame=(0, t_end-1)):
        plt.style.use('seaborn-white')
        plt.title("%d step"%(frame),y=1.1)
        plt.contourf(get_configuration(data, frame),10,cmap='Spectral') # n equally spaced lines
        plt.colorbar()
        #plt.clim(0,1)
        plt.show()
    return interact(_show)

def get_configuration(data, t):
    """Get configuration from data at a given time"""
    return convert_to_df(np.asarray(data.iloc[t]).reshape(size,size))


# In[172]:


def time_ev_mat(grid,t_end=500):
    """Main of the simulation, returns path and name of the output file"""


    start_t = time.clock()
    t = 0  
    size = np.shape(grid)[0]
    a = np.zeros(shape=[4,size,size])
    Id_mat = np.eye(size)
    full1_mat = np.full([size,size],1)
    
    result_path_name = create_size_folderdata(size)
    
    filePath = result_path_name+"/"+name_data(t_end)
 
    #to avoid overwrite
    if os.path.exists(filePath):
        os.remove(filePath)
    
    print("\n")
    
    grid = convert_to_df(grid)
    for i in range(0,size):
        for j in range(0,size):
            a[0,i,j] = r1(i,j)
            a[1,i,j] = r2(i,j)
            a[2,i,j] = r3(i,j) 
            a[3,i,j] = r4(i,j) 

    a_df=[]
    a_df.append(convert_to_df(a[0,:,:]))
    a_df.append(convert_to_df(a[1,:,:]))
    a_df.append(convert_to_df(a[2,:,:]))
    a_df.append(convert_to_df(a[3,:,:]))
    
    t_flag=5
    while t<t_end+1: #not to lose last step
        #running flag
        if(t%int(t_end/t_flag))==0:
            print("Process @%d/%d (%.2f/100)"%(t,t_end,(100*(t)/t_end)))
            print("Normalized sum of grid %.8f"%(sum(sum(grid.values))/(size*size)))
            print("Elapsed time = %.3fseconds\n"%(time.clock()-start_t))    
            #not to lose first step
        #here configurations are saved as 1D arrays, one per each time step
        (convert_to_df(grid.values.flatten()).T).to_csv(result_path_name+"/"+name_data(t_end),header=None,index=None,sep=",",float_format='%f',mode='a')

        #working directly on matrices to avoid nested for loops with plenty fo evaluations
        
        shifted_distrib = get_distrib_mat(grid)
        N = shifted_distrib[0]
        E = shifted_distrib[1]
        S = shifted_distrib[2]
        W = shifted_distrib[3]
        
        increment = np.zeros(shape=[size,size])
        increment = increment + (W-grid).values*grid.values*a_df[0].values
        increment = increment + (E-grid).values*grid.values*a_df[1].values
        increment = increment + (N-grid).values*grid.values*a_df[2].values
        increment = increment + (S-grid).values*grid.values*a_df[3].values
        
        #implement mask for thresholds
        
        
        grid = grid + increment
        t+=1

    end_t = time.clock()
    print("\n\nDone in %.3fseconds"%(end_t-start_t))  
    file = open(result_path_name+"/"+"Stats.txt","w") 
 
    file.write("Done in %.3fseconds"%(end_t-start_t)) 
    
    return result_path_name+"/"+name_data(t_end)


# In[ ]:





# In[34]:


#shift order to simulate test or paper configuration
size = 300
std = 266/2
center = [133,133]

size = 100
std = size/5
center = [size/8,size/2]




t_end = 100

grid = convert_to_df(makeGaussian(size=size,center=center, fwhm = std))
plt.matshow(grid)
plt.title("Initial Configuration",y=1.1)
plt.show()
output_path = time_ev_mat(grid,t_end=t_end) #execute and returns path of data output 


# In[35]:


#display_sequence_contourf(output_path,t_end)


# In[110]:


data = get_data(output_path,t_end)


# In[171]:


fig, axes = plt.subplots(nrows=1, ncols=4,figsize=(20,5))
ncolors = 15
cmap ="jet"
fig.suptitle("Configurations for %d x %d grid, $(x_{0},y_{0})=(%d,%d)$, $\\sigma=%d, t_{end}=%d$"%(size,size,center[0],center[1],std,t_end),fontsize=20,y=1.)

im0 = axes[0].contourf(get_configuration(data, 1),ncolors,cmap=cmap) # n equally spaced lines
im1 = axes[1].contourf(get_configuration(data, int(t_end/2)),ncolors,cmap=cmap) # n equally spaced lines
im2 = axes[2].contourf(get_configuration(data, int(t_end*3/4)),ncolors,cmap=cmap) # n equally spaced lines
im3 = axes[3].contourf(get_configuration(data, int(t_end)),ncolors,cmap=cmap) # n equally spaced lines

#titles

axes[0].set_title('T = 0',fontsize=16)
axes[1].set_title('T = $\\frac{%d}{%d}\,\,t_{end}$'%(1,2),fontsize=16)
axes[2].set_title('T = $\\frac{%d}{%d}\,\,t_{end}$'%(3,4),fontsize=16)
axes[3].set_title('T = $t_{end}$',fontsize=16)

#labels
axes[0].set_ylabel("B species",fontsize=16)
axes[0].set_ylabel("B species",fontsize=16)
axes[0].set_xlabel("A species",fontsize=16)
axes[1].set_xlabel("A species",fontsize=16)
axes[2].set_xlabel("A species",fontsize=16)
axes[3].set_xlabel("A species",fontsize=16)

fig.subplots_adjust(right=0.8,top=.85)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im0, cax=cbar_ax)
plt.savefig(create_size_folderdata(size)+"/"+"t_end=%d.png"%(t_end))
plt.show()
