using Pkg
using Plots
using LinearAlgebra
using CSV, DataFrames
############ 2D, not optimized and a bit too much flexible  ############

function mat_mul(mat1::Array{Float64,2}, mat2::Array{Float64,2})
    return mat1 .*mat2
end

function distrib_0(position,state_vec_0,stdev)#,still 2d to implement in multidim
    exponent=0.
    for i in 1:2
        exponent+= (position[i]-state_vec_0[i])^2
    end
    return exp(-exponent/stdev)
end

function reaction_weight(reaction_index)
    c1=c4 = 3e3
    c2=c5 = 1.1e4
    c3=c6 = 1e-3
    β = γ = 2

    if reaction_index==1
        newfunction = function (x,y) return c1/(c2+(y^β)) end
    end
    if reaction_index==2
        newfunction = function (x,y) return c3*x end
    end
    if reaction_index==3
        newfunction = function (x,y) return c4/(c5+(x^γ)) end
    end
    if reaction_index==4
        newfunction = function (x,y) return c6*y end
    end
    return newfunction
    #reaction_weight(i)(j,k) #use like this: i= index of reaction, (j,k)=(x,y)
end

function initial_grid(size,initial_cond,std)
    grid = zeros(size[1],size[2])
    for i in 1:size[1]
        for j in 1:size[2]
            grid[i,j] = distrib_0([i,j],[initial_cond[1],initial_cond[2]],std)
        end
    end
    return grid
end

grid_size = 300

#Can be optimized with only one tensor, but so far is still okay
a1 = zeros(grid_size,grid_size,5) #1=itself; 2,3,4,5=N,S,E,W
a2 = zeros(grid_size,grid_size,5)
a3 = zeros(grid_size,grid_size,5)
a4 = zeros(grid_size,grid_size,5)

for i in 1:grid_size
    for j in 1:grid_size

        #Propensities

        #First reaction
        a1[i,j,1] = reaction_weight(1)(i,j)
        a1[i,j,2] = reaction_weight(1)(i-1,j)
        a1[1,j,2] = 0 #to consider the "out of grid", look N
        a1[i,j,3] = reaction_weight(1)(i,j+1)
        a1[i,grid_size,3] = 0 # E
        a1[i,j,4] = reaction_weight(1)(i+1,j)
        a1[grid_size,j,4] = 0 # S
        a1[i,j,5] = reaction_weight(1)(i,j-1)
        a1[i,1,5] = 0 # W

        #Second Reaction
        a2[i,j,1] = reaction_weight(2)(i,j)
        a2[i,j,2] = reaction_weight(2)(i-1,j)
        a2[1,j,2] = 0 #to consider the "out of grid", look N
        a2[i,j,3] = reaction_weight(2)(i,j+1)
        a2[i,grid_size,3] = 0 # E
        a2[i,j,4] = reaction_weight(2)(i+1,j)
        a2[grid_size,j,4] = 0 # S
        a2[i,j,5] = reaction_weight(2)(i,j-1)
        a2[i,1,5] = 0 # W

        #Third Reaction
        a3[i,j,1] = reaction_weight(3)(i,j)
        a3[i,j,2] = reaction_weight(3)(i-1,j)
        a3[1,j,2] = 0 #to consider the "out of grid", look N
        a3[i,j,3] = reaction_weight(3)(i,j+1)
        a3[i,grid_size,3] = 0 # E
        a3[i,j,4] = reaction_weight(3)(i+1,j)
        a3[grid_size,j,4] = 0 # S
        a3[i,j,5] = reaction_weight(3)(i,j-1)
        a3[i,1,5] = 0 # W

        #Fourth Reaction
        a4[i,j,1] = reaction_weight(4)(i,j)
        a4[i,j,2] = reaction_weight(4)(i-1,j)
        a4[1,j,2] = 0 #to consider the "out of grid", look N
        a4[i,j,3] = reaction_weight(4)(i,j+1)
        a4[i,grid_size,3] = 0 # E
        a4[i,j,4] = reaction_weight(4)(i+1,j)
        a4[grid_size,j,4] = 0 # S
        a4[i,j,5] = reaction_weight(4)(i,j-1)
        a4[i,1,5] = 0 # W
    end
end

#Plot propensities
heatmap(a1[:,:,1], c = :viridis)
heatmap(a2[:,:,1], c = :viridis)
heatmap(a3[:,:,1], c = :viridis)
heatmap(a4[:,:,1], c = :viridis)

#Plot sum of propensities
heatmap(a1[:,:,1] + a2[:,:,1] + a3[:,:,1] + a4[:,:,1], c = :viridis)

sizes = [grid_size,grid_size]
std = 266*3
@time begin
    let grid_run = initial_grid(sizes,[133,133],std)
        grid0 = grid_run
        # Start of time loop
        print("\n\n\n\nStarting now!")
        t = 0
        t_end = 100000 
        dt = .1
        Plots.display(heatmap(grid0,title=string(dt*t),c = :cividis))

        while t<t_end/dt
            t+=1
            if t % trunc(Int,(t_end/dt)/5000)==0
               print("Process @ ",t,"/",t_end/dt,". Effective t = ",dt*t,"\n")
               print("Total prob = ",sum(grid_run),"\n\n")
               Plots.display(heatmap(grid_run,title=string(dt*t),c = :cividis))
            end

            #evaluate shifted grid, always NESW syntax
            grid_N = zeros(grid_size,grid_size)
            grid_E = zeros(grid_size,grid_size)
            grid_S = zeros(grid_size,grid_size)
            grid_W = zeros(grid_size,grid_size)
            for j in 1:grid_size
                try
                    grid_N[j,:] = grid_run[j-1,:] #N
                catch err
                    grid_N[j,:]=zeros(grid_size)
                end
                try
                    grid_E[:,j] = grid_run[:,j+1] #E
                catch err
                    grid_E[:,j]=zeros(grid_size)
                end
                try
                    grid_S[j,:] = grid_run[j+1,:] #S
                catch err
                    grid_S[j,:]=zeros(grid_size)
                end
                try
                    grid_W[:,j] = grid_run[:,j-1] #W
                catch err
                    grid_W[:,j]=zeros(grid_size)
            end
        end

        increment = zeros(grid_size,grid_size)
        increment += dt*(   a1[:,:,5].*grid_W - a1[:,:,1].*grid_run +
                            a2[:,:,3].*grid_E - a2[:,:,1].*grid_run +
                            a3[:,:,2].*grid_N - a3[:,:,1].*grid_run +
                            a4[:,:,4].*grid_S - a4[:,:,1].*grid_run    )
        grid_run+=increment
        end
        CSV.write("final_config.csv",  DataFrame(grid_run), writeheader=false)
    end
end

