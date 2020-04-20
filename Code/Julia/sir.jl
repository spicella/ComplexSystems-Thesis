using Pkg
#Pkg.add("Plots")
using Plots
#Pkg.add("DifferentialEquations")
#Pkg.add("SparseArrays")
#Pkg.add("LinearAlgebra")
#Pkg.add("CSV")
#Pkg.add("DataFrames")
#Pkg.add("Formatting")
#Pkg.add("StaticArrays") #https://docs.sciml.ai/v5.0.0/tutorials/ode_example.html
#Pkg.add("Profile")
using Profile
using CSV, DataFrames
using Formatting
using StaticArrays
using DifferentialEquations
#Pkg.test("DifferentialEquations")
using SparseArrays
using LinearAlgebra
print("Finished importing packages\n\n")

#Initial Configuration
N=50
#N=300
mu = [N*.1 N*.9]
sigma = [N/10 N/10]

#set current main directory
cd(dirname(@__FILE__))
main_path = pwd()
print("pwd:\n")
print(main_path)
try
      mkdir(main_path*"/data")
catch e
     println("\nData folder already there!\n")
end
s = format("/{:,d}_{:,d}_{:,d}",N,mu[1],mu[2]) ;
try
      mkdir(main_path*"/data"*s)
catch e
     println("\nConfig folder already there!\n")
end
data_path = main_path*"/data"*s
print(data_path)
#print(Pkg.installed())

###############################
#                             #
#        Second Attempt       #
#                             #
###############################

function config0(N,μ,σ)
      plot_grid = zeros(Float64,N,N)
      plot_grid = sparse(plot_grid)
      for i in 1:N
            for j in 1:N
                  plot_grid[i,j]=1/( 2 * pi * sqrt(σ[1] * σ[2]))*
                  exp(-(i-μ[1] )^2 / 2 / σ[1] )*
                  exp(-(j-μ[2] )^2 / 2 / σ[2] )
            end
      end

      return plot_grid
end

grid0 = config0(N,mu,sigma)
print("\nN-mu-sigma\n")
print(N)
print("\n")
print(mu)
print("\n")
print(sigma)
print("\n")

#save input
#df_grid0 = DataFrame(grid0)
#s = format("/N={:,d}_mu=[{:,.2f},{:,.2f}]_sigma=[{:,.2f},{:,.2f}]_grid0.dat", N,mu[1],mu[2],sigma[1],sigma[2]) ;
#CSV.write(data_path*s,df_grid0,writeheader=false)

grid0 = reshape(sparse(grid0), (N^2))

#Start loop over R0
r0_range = range(0; stop = 4, step = .4)
len_r0_range = length(r0_range)

#Compute J
print("\n\n-----------------CME-----------------\n\n")
for i in 1:len_r0_range
      print("\n-------------$i out of $len_r0_range-------------\n")
      J=nothing
      p = [r0_range[i] 1.] #beta, gamma so that p[1]/p[2] = R0
      #Compute J
      Id = sparse(Matrix{Float64}(I, N, N))

      E_1 = sparse(zeros(Float64,N,N))
      Em1 = sparse(zeros(Float64,N,N))

      for i in 1:N-1
            E_1[i+1,i] = 1
            Em1[i,i+1] = 1
      end


      T01 = sparse(kron( Id, E_1 ))
      T10 = sparse(kron( E_1, Id ))
      T0m1 = sparse(kron( Id, Em1 ))
      Tm10 = sparse(kron( Em1, Id ))

      Id, E_1,Em1 = nothing, nothing, nothing

      #Plots.heatmap(Tm10)

      TWx1 = sparse(zeros(Float64,N^2,N^2))
      TWx2 = sparse(zeros(Float64,N^2,N^2))


      for i in 1:N
            for j in 1:N
                  idx1 = j+(i-1)*N
                  TWx1[idx1,idx1] = i #X
                  TWx2[idx1,idx1] = j #Y
            end
      end

      #Plots.heatmap(TWx1)
      #Plots.heatmap(TWx2)
      #Plots.heatmap(Matrix(Tm10*TWx1))
      print("Initializing J!\n")

      J = sparse(zeros(Float64,N^2,N^2))

      J =   (p[1]/N)*(Tm10*(TWx1*TWx2) - TWx1*TWx2) +    #reaction1: decrease of S
            (p[1]/N)*(T01*(TWx1*TWx2) - TWx1*TWx2) +     #reaction2: increase of I
            p[2]*(T0m1*TWx2-TWx2)                           #reaction3: decrease of I
      Tm10,T01,TWx1 = nothing, nothing,nothing
      T0m1,T10,TWx2 = nothing, nothing,nothing
      #gc

      print("J initialized, now solving...\n")

      tspan = (0.0,5.0e2)
      f(u,p,t) = J*u
      prob = ODEProblem(f,grid0,tspan)
      #sol = solve(prob,Vern7(),abstol=1e-100,reltol=3e-14)
      sol = solve(prob,Feagin12(),abstol=1e-100,reltol=3e-14)
      #sol = solve(prob,Vern7())
      sol.t
      configs = reshape(sol.u[1],N,N),reshape(sol.u[end],N,N)
      configs = Matrix(configs[1]),Matrix(configs[end])

      df_grid_final = DataFrame(configs[end])
      s = format("/{:,.5e}.dat",p[1]) ;
      CSV.write(data_path*s,df_grid_final,writeheader=false)

      print("...computation ended!\n")
      print("R0 = ",p[1])
      print("\n")

end

print("\n\n-----------------ODE-----------------\n\n")
#ODE
function sir(du,u,p,t)
 du[1] = -(p[1]/N)*u[1]*u[2]
 du[2] = (p[1]/N)*u[1]*u[2]-p[2]*u[2]
end

tspan = (0.0,5.0e2)

#Initialize output
df = DataFrame(p1 = Float64[], x_eq = Float64[], y_eq = Float64[])

print("Start solving now\n")
for i in 1:len_r0_range
    s = format("{:,d} out of {:,d}\n", i,len_r0_range)
    print(s)
    p = [r0_range[i] 1.] #beta, gamma so that p[1]/p[2] = R0
    #Solve
    prob = ODEProblem(sir, mu, tspan,p)
    sol = solve(prob,Feagin12(),abstol=1e-100,reltol=3e-14)
    #Save
    push!(df, [p[1] sol[end][1] sol[end][2]])
end
print("Finished, now saving\n")
s = format("/N={:,d}_mu=[{:,.2f},{:,.2f}]_analytical.dat", N,mu[1],mu[2]) ;
CSV.write(data_path*s,df,writeheader=false)

print("\n\nAll done!")
