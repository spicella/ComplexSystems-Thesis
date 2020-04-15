using Pkg
#Pkg.add("Plots")
using Plots
#Pkg.rm("DifferentialEquations")
#Pkg.add("DifferentialEquations")
#Pkg.add("Plots")
#Pkg.add("SparseArrays")
#Pkg.add("LinearAlgebra")
#Pkg.test("DifferentialEquations")
using DifferentialEquations
using SparseArrays
using LinearAlgebra

#print(Pkg.installed())

#SIR first attempt
#
# function config0(N,μ,σ)
#       grid = zeros(Float64,N)
#       for i in 1:N
#             grid[i]=1/( 2 * pi * sqrt(σ))*
#            exp(-(i-μ)^2 / 2 / σ )
#       end
#       return grid
# end
#
# β,γ = 10.,1.   #R0=β/γ
# N = 6 #grid resolution
#
# Id = Matrix{Float64}(I, N, N)
# T1 = zeros(Float64,N,N)
# T1m = zeros(Float64,N,N)
#
# for i in 1:N-1
#       T1m[i,i+1] = 1 #takes position +1
#       T1[i+1,i] = 1 #takes position -1
# end
#
# #Here before running main of the code
# #some garbage collector must be added
#
# #Example
# x0 = config0(N,N/2,N/2)
# y0 = config0(N,N,N/2)
#
# Plots.plot(x0)
# Plots.plot!(y0)
#
# Jx = (T1m-Id)*β/N
# Jy1 = (T1-Id)*β/N
# Jy2 = γ*(T1m-Id)
# #
# # Jx = (T1m)*β/N
# # Jy1 = (T1)*β/N
# # Jy2 = γ*(T1m)
#
# Jtot(x,y,p,t) = [Jx*(x.*y) Jy1*(x.*y)+Jy2*y]
# #      du[1] = Jon_on*u[1]+Jon_off*u[2]
# #      du[2] = Joff_on*u[1]+Joff_off*u[2]
#
# tspan = (0.0,1.0e3)
# prob = ODEProblem(Jtot,[x0 y0],tspan,[])
# sol = solve(prob,Vern7())
# #sol = solve(prob,Vern7(),abstol=1e-50,reltol=3e-30)
#
# Plots.plot(sol.u[end][:,1])
# Plots.plot!(sol.u[end][:,2])
# Plots.plot!(sol.u[end][:,1].+sol.u[end][:,2])
#
# sol.u[end][:,2]==y0
#
# sol.t
#
# sol
#
#
# Jtot(x0,y0,[],100)[:,1]
#


###############################
#                             #
#        Second Attempt       #
#                             #
###############################

N=50

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

grid0 = config0(N,[N/3,N/1.5],[N/5,N/2])[2]
Plots.heatmap(grid0)

grid0 = reshape(sparse(grid0), (N^2))

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

#Plots.heatmap(Tm10)

TWx1 = zeros(Float64,N^2,N^2)
TWx2 = zeros(Float64,N^2,N^2)


for i in 1:N
      for j in 1:N
            idx1 = j+(i-1)*N
            TWx1[idx1,idx1] = i #X
            TWx2[idx1,idx1] = j #Y
      end
end

#Plots.heatmap(TWx1)

TWx1 = sparse(TWx1)
TWx2 = sparse(TWx2)
T01 = sparse(T01)
T10 = sparse(T10)
T0m1 = sparse(T0m1)
Tm10 = sparse(Tm10)
#Plots.heatmap(TWx2)
#Plots.heatmap(Matrix(Tm10*TWx1))

p = [2 1] #beta, gamma

J =   (p[1]/p[2])*(Tm10*(TWx1*TWx2) - TWx1*TWx2) +    #reaction1: decrease of S
      (p[1]/p[2])*(T01*(TWx1*TWx2) - TWx1*TWx2) +     #reaction2: increase of I
      p[2]*(T0m1*TWx2-TWx2)                           #reaction3: decrease of I
Tm10,T01,TWx1 = nothing, nothing,nothing
T0m1,T10,TWx2 = nothing, nothing,nothing
#gc

print("Preliminary finished, ready for computing")


tspan = (0.0,1)
f(u,p,t) = J*u
prob = ODEProblem(f,grid0,tspan)
#sol = solve(prob,Vern7(),abstol=1e-100,reltol=3e-14)
#sol = solve(prob,Vern7(),abstol=1e-8,reltol=3e-6)
sol = solve(prob,Vern7())
final_config = reshape(sol.u[end],N,N)
final_config
replace!(final_config, NaN=>0)
