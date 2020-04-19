using Pkg
#Pkg.add("Plots")
using Plots
#Pkg.add("DifferentialEquations")
#Pkg.add("SparseArrays")
#Pkg.add("LinearAlgebra")
#Pkg.add("CSV")
#Pkg.add("DataFrames")
#Pkg.add("Formatting")
using CSV, DataFrames
using Formatting
using DifferentialEquations
#Pkg.test("DifferentialEquations")
using SparseArrays
using LinearAlgebra

#set current directory
cd(dirname(@__FILE__))
main_path = pwd()
print("pwd:")
print(main_path)
try
      mkdir(main_path*"/data")
catch e
     println("Folder already there!")
end
data_path = main_path*"/data"
#print(Pkg.installed())

###############################
#                             #
#        Second Attempt       #
#                             #
###############################

N=300

#meshgrid
function config0(N,μ,σ)
      grid = zeros(Float64,N^2,N^2)
      plot_grid = zeros(Float64,N,N)
      for i in 1:N
            for j in 1:N
                  plot_grid[i,j]=1/( 2 * pi * sqrt(σ[1] * σ[2]))*
                  exp(-(i-μ[1] )^2 / 2 / σ[1] )*
                  exp(-(j-μ[2] )^2 / 2 / σ[2] )
            end
      end
      Plots.heatmap!(plot_grid)
      for i in 1:N^2
            for j in 1:N
                  for k in 1:N
                        grid[i,i]=plot_grid[j,k]
                  end
            end
      end
      return grid, plot_grid
end
mu = [N*.1 N*.9]
sigma = [N/10 N/10]
grid0 = config0(N,mu,sigma)[2]
Plots.heatmap(grid0)

#save input
df_grid0 = DataFrame(grid0)
s = format("/N={:,d}_mu=[{:,.2f},{:,.2f}]_sigma=[{:,.2f},{:,.2f}]_grid0.dat", N,mu[1],mu[2],sigma[1],sigma[2]) ;
CSV.write(data_path*s,df_grid0,writeheader=false)

grid0 = reshape(sparse(grid0), (N^2))

#Start loop over R0
r0_range = range(0; stop = 4, step = .2)
len_r0_range = length(r0_range)

Id = Matrix{Float64}(I, N, N)
Id = sparse(Id)

E_1 = zeros(Float64,N,N)
Em1 = zeros(Float64,N,N)

for i in 1:N-1
      E_1[i+1,i] = 1
      Em1[i,i+1] = 1
end
E_1 = sparse(E_1)
Em1 = sparse(Em1)

T01 = kron( Id, E_1 )
T10 = kron( E_1, Id )

T0m1 = kron( Id, Em1 )
Tm10 = kron( Em1, Id )
Id, E_1,Em1 = nothing, nothing, nothing


TWx1 = zeros(Float64,N^2,N^2)
TWx2 = zeros(Float64,N^2,N^2)


for i in 1:N
      for j in 1:N
            idx1 = j+(i-1)*N
            TWx1[idx1,idx1] = i #X
            TWx2[idx1,idx1] = j #Y
      end
end

TWx1 = sparse(TWx1)
TWx2 = sparse(TWx2)
T01 = sparse(T01)
T10 = sparse(T10)
T0m1 = sparse(T0m1)
Tm10 = sparse(Tm10)

#Compute J
for i in 1:len_r0_range
      print("\n-------------$i out of $len_r0_range-------------\n")
      p = [r0_range[i] 1.] #beta, gamma so that p[1]/p[2] = R0

      J =   (p[1]/N)*(Tm10*(TWx1*TWx2) - TWx1*TWx2) +    #reaction1: decrease of S
            (p[1]/N)*(T01*(TWx1*TWx2) - TWx1*TWx2) +     #reaction2: increase of I
            p[2]*(T0m1*TWx2-TWx2)                           #reaction3: decrease of I

      #garbage collector
      Tm10,T01,TWx1 = nothing, nothing,nothing
      T0m1,T10,TWx2 = nothing, nothing,nothing


      print("Preliminary finished, ready for computing\n")

      tspan = (0.0,2.0e2)
      f(u,p,t) = J*u
      prob = ODEProblem(f,grid0,tspan)
      #ODE113 Matlab ==>RK12, Vern7 is insufficient
      #sol = solve(prob,Vern7(),abstol=1e-100,reltol=3e-14)
      sol = solve(prob,CVODE_Adams(),abstol=1e-100,reltol=3e-14)
      #sol = solve(prob,Vern7())
      configs = reshape(sol.u[end],N,N)
      configs = Matrix(configs[end])

      df_grid_final = DataFrame(configs[end])
      s = format("/N={:,d}_mu=[{:,.2f},{:,.2f}]_sigma=[{:,.2f},{:,.2f}]_grid_final_R0={:,.2f}_.dat", N,mu[1],mu[2],sigma[1],sigma[2],p[1]) ;
      CSV.write(data_path*s,df_grid_final,writeheader=false)

      print("Computation ended\n")
      print("R0 = ",p[1])
      print("\n")

end

print("\n\nAll done!")
