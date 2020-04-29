using Pkg
#Pkg.add("Plots")
#Pkg.add("PyPlot")
#Pkg.add("ODEInterfaceDiffEq")
#Pkg.add("DifferentialEquations")
#Pkg.add("SparseArrays")
#Pkg.add("LinearAlgebra")
#Pkg.add("CSV")
#Pkg.add("DataFrames")
#Pkg.add("Formatting")
#Pkg.add("StaticArrays") #https://docs.sciml.ai/v5.0.0/tutorials/ode_example.html
#Pkg.add("Profile")
#Pkg.test("")
#Pkg.rm("")
using Profile
using CSV, DataFrames
using Formatting
using DifferentialEquations
using SparseArrays
using LinearAlgebra
using Plots
pyplot()
print("Finished importing packages\n\n")

#Initial Configuration
N=50
mu = [N*.1 N*.9]
sigma = [N/10 N/10]

#set current main directory
cd(dirname(@__FILE__))
main_path = pwd()
print("pwd:\n")
print(main_path)
try
      mkdir(main_path*"/data")
      println("\nCreated data folder\n")
catch e
     println("\nData folder already there!\n")
end
s = format("/{:,d}_{:,d}_{:,d}",N,mu[1],mu[2]) ;
try
      mkdir(main_path*"/data"*s)
      println("\nCreated results folder\n")
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

#Display initial configuration
#start_plot = Plots.heatmap(grid0)
#display(start_plot)
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

grid0 = reshape(grid0, (N^2))

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
      Id = sparse(Matrix{Float64}(I, N^2, N^2))
      #J =   p[1]/N*(Tm10*(TWx1*TWx2) - TWx1*TWx2) +    #reaction1: decrease of S
      #      p[1]/N*(T01*(TWx1*TWx2) - TWx1*TWx2) +    #reaction2: increase of I
      #      p[2]*(T0m1*TWx2-TWx2)                     #reaction3: decrease of I

      J =   p[1]/N*(Tm10*T01-Id)* TWx2 * TWx1 + #(S,I)->(S-1,I+1)
            p[2]*(T0m1-Id)* TWx2                #
      Tm10,T01,TWx1 = nothing, nothing,nothing
      T0m1,T10,TWx2 = nothing, nothing,nothing
      Id = nothing
      #gc
      J = Matrix(J) #sparse to normal matrix
      print("J initialized, now solving...\n")

      tspan = (0.0,1.0e2)
      function f(du,u,p,t)
          du .= J*u
      end
      #f(u,p,t) = J*u

      prob = ODEProblem(f,grid0,tspan)

      sol = solve(prob,Vern7(),progress=true,save_start=false,save_on=false,abstol=1e-100,reltol=3e-14)

      #sol = solve(prob,RadauIIA5(),progress=true,save_start=false,save_on=false,abstol=1e-100,reltol=3e-14)
      #sol = solve(prob,Vern7(),save_on=false,save_start=false,save_end=true,abstol=1e-100,reltol=3e-14)
      #sol = solve(prob,RadauIIA5(),save_on=false,abstol=1e-100,reltol=3e-14)
      #sol = solve(prob, RadauIIA5(),save_on=false,abstol=1e-100,reltol=3e-14)

      #sol.t
      print("Solving completed, now saving\n")
      print("Reshaped")
      last_config = reshape(sol.u[end],N,N)
      #Display configs live
      #final_plot = Plots.heatmap(configs[end])
      #display(final_plot)
      last_config = convert(DataFrame, last_config)
      s = format("/{:,.5e}.dat",p[1]) ;
      CSV.write(data_path*s,last_config,writeheader=false)

      print("...computation ended!\n")
      print("R0 = ",p[1])
      print("\n")

end

print("\n\n-----------------ODE-----------------\n\n")
#ODE
function sir(du,u,p,t)
 du[1] = -p[1]*u[1]*u[2]
 du[2] =  p[1]*u[1]*u[2]-p[2]*u[2]
end

tspan = (0.0,1.0e2)

#Initialize output
df = DataFrame(p1 = Float64[], x_eq = Float64[], y_eq = Float64[])

print("Start solving now\n")
for i in 1:len_r0_range
    s = format("{:,d} out of {:,d}\n", i,len_r0_range)
    print(s)
    p = [r0_range[i] 1.] #beta, gamma so that p[1]/p[2] = R0
    #Solve
    prob = ODEProblem(sir, mu, tspan,p)
    #sol = solve(prob,Feagin12(),abstol=1e-100,reltol=3e-14)
    #sol = solve(prob,Vern7(),save_on=false,save_start=false,save_end=true,abstol=1e-100,reltol=3e-14)
    sol = solve(prob,Vern7(),abstol=1e-100,reltol=3e-14)
    #Save
    push!(df, [p[1] sol[end][1] sol[end][2]])
end
print("Finished, now saving\n")
s = format("/N={:,d}_mu=[{:,.2f},{:,.2f}]_analytical.dat", N,mu[1],mu[2]) ;
CSV.write(data_path*s,df,writeheader=false)

print("\n\n---------------All done!---------------")
print("\nN-mu-sigma\n")
print(N)
print("\n")
print(mu)
print("\n")
print(sigma)
print("\n")
print("\n\n---------------------------------------")
