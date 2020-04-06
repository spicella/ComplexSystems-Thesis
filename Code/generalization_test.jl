using Pkg
using Plots
using DifferentialEquations
using LinearAlgebra

#Example for generalization of problem
#using the self-regulating gene case
#procedure is working, just need to plug
#in the correct coefficients

function config0(N,μ,σ)
      grid = zeros(Float32,N)
      for i in 1:N
            grid[i]=1/( 2 * pi * sqrt(σ))*
           exp(-(i-μ)^2 / 2 / σ )
      end
      return grid
end


function h_func(x)
      return h_val/2 * (x-1) *(x-2) #as Julia has 1 as first index..
end

#Plots.plot(config0(100,20,10))
g_on, g_off, k , h_val, f = 1,1,.005,1,1

N = 100

Id = zeros(Float32,N,N)
T1 = zeros(Float32,N,N)
T1m = zeros(Float32,N,N)
Wx0 = zeros(Float32,N,N)
Wh = zeros(Float32,N,N)
Wf = zeros(Float32,N,N)
Wx = zeros(Float32,N,N)
Wx0[1,1]=1
for i in 1:N
      Id[i,i] = 1
      Wh[i,i] = 1*h_func(i)
      Wf[i,i] = 1*f
      Wx[i,i] = i
end

for i in 1:N-1
      T1m[i+1,i] = 1
      T1[i,i+1] = 1
end

Jon_on = g_on*(T1-Id)+k*(T1m-Id)*Wx-Wh
Jon_off = Wf + k*Wx0*T1m

Joff_on = g_off*(T1-Id)-Wf
Joff_off = k*(T1m-Id)*Wx + Wh

#Here before running main of the code
#some garbage collector must be added


#Example
uon0 = config0(N,N/3,N/2)
uoff0 = config0(N,N/5,N/2)


func(u,p,t) = [Jon_on*uon+Jon_off*uoff Joff_on*uon+Joff_off*uoff]
#      du[1] = Jon_on*u[1]+Jon_off*u[2]
#      du[2] = Joff_on*u[1]+Joff_off*u[2]

tspan = (0.0,100.0)
prob = ODEProblem(func,[uon0 uoff0],tspan)
sol = solve(prob,Vern7(),abstol=1e-8,reltol=3e-6)

#Second variable plot
Plots.plot(sol[length(sol)][:,2])

#Then, a meshgrid must be created..
