using Pkg
using Plots
using DifferentialEquations
using LinearAlgebra

function config0(N,μ,σ)
      grid = zeros(Float32,N^2,N^2)
      plot_grid = zeros(Float32,N,N)
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

#Plots.heatmap(config0(30,[10,10],[20,5])[2])

#Parameters
c1,c2,c3,c4,c5,c6,β,γ=3e3,1.1e4,1e-3,3e3,1.1e4,1e-3,2,2 #from paper
function weight_matrices(N)
      react1 = function (y) return c1/(c2+(y^β)) end
      react3 = function (x) return c4/(c5+(x^γ)) end

      TWx1 = zeros(Float32,N^2,N^2)
      TWox1 = zeros(Float32,N^2,N^2)
      TWx2 = zeros(Float32,N^2,N^2)
      TWox2 = zeros(Float32,N^2,N^2)

      for i in 1:N
            for j in 1:N
                  idx1 = j+(i-1)*N

                  TWox2[idx1,idx1] = react1(j)
                  TWox1[idx1,idx1] = react3(i)

                  TWx1[idx1,idx1] = i #second reaction
                  TWx2[idx1,idx1] = j #fourth reaction
            end
      end

      return TWx1, TWox1, TWx2, TWox2
end

N = 300
W_mats = weight_matrices(N)
TWx1, TWox1, TWx2, TWox2 = W_mats[1], W_mats[2], W_mats[3], W_mats[4]
W_mats = nothing

#Plots.heatmap(TWx2)

##
E_1 = zeros(Float32,N,N)
Em1 = zeros(Float32,N,N)
Id = zeros(Float32,N,N)

for i in 1:N-1
      E_1[i+1,i] = 1
      Em1[i,i+1] = 1
end
for i in 1:N
      Id[i,i] = 1
end


T01 = kron(Id,E_1)
T10 = kron(E_1,Id)
#Plots.heatmap(T01)
#Plots.heatmap(T10)
T0m1 = kron(Id,Em1)
Tm10 = kron(Em1,Id)

Id=nothing
Em1=nothing
E_1=nothing

J = zeros(Float32,N^2,N^2)

J= T10 * TWox2
T10=nothing

J.+= c3 .*(Tm10 * TWx1)
Tm10=nothing

J.+=  c3 .* TWx1
TWx1=nothing

J.+= T01 * TWox1
T01=nothing

J.+= c6 .*(T0m1 * TWx2)
T0m1=nothing

J.+= c6 .* TWx2
TWx2=nothing

J = J.-TWox2.-TWox1

TWox1=nothing
TWox2=nothing
#J =   T10 * TWox2  .+ c3 *  Tm10 * TWx1 .- TWox2 .- c3 * TWx1.+
#      .+   T01 * TWox1  .+ c6 *  T0m1 * TWx2 .- TWox1 .- c6 * TWx2

Plots.heatmap(J)

#MatLab: ode113 => Julia: Vern7()
grid0 = config0(N,[N/3,N/6],[N,N/2])[2]
grid0 = config0(300,[133,133],[133,133])[2]

Plots.heatmap(grid0)
grid_diag = diag(grid0)

grid_1darray = reshape(grid0,N^2,1)
J*grid_1darray

u0 = grid_1darray
tspan = (0.0,1e5)
f(u,p,t) = J*u
prob = ODEProblem(f,u0,tspan)
#sol = solve(prob,Vern7(),abstol=1e-100,reltol=3e-14)
#sol = solve(prob,Vern7(),abstol=1e-8,reltol=3e-6)
sol = solve(prob,Vern7())

length(sol)

grid_out = reshape(sol[length(sol)],N,N)
Plots.heatmap(grid_out)
