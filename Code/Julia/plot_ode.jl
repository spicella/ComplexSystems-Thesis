using DifferentialEquations, Plots
using DelimitedFiles, CSV, DataFrames
using Formatting

#Paths
cd(dirname(@__FILE__))
main_path = pwd()
print("pwd:")
print(main_path)
#mkdir(main_path*"/data")
data_path = main_path*"/data"

#ODE
function sir(du,u,p,t)
 du[1] = -(p[1]/N)*u[1]*u[2]
 du[2] = (p[1]/N)*u[1]*u[2]-p[2]*u[2]
end

#Initial Conditions
N=300
u0 = [N/4 N/4]
tspan = (0., 5.0e0)

#Initialize output
df = DataFrame(p1 = Float64[], x_eq = Float64[], y_eq = Float64[])
#Initialize range
r0_range = range(.3; stop = 5.2, step = .25)
len_r0_range = length(r0_range)

for i in 1:len_r0_range
    p = [r0_range[i] 1.] #beta, gamma so that p[1]/p[2] = R0
    #Solve
    prob = ODEProblem(sir, u0, tspan,p)
    sol = solve(prob,Vern7(),abstol=1e-100,reltol=3e-14)
    #Save
    push!(df, [p[1] sol[end][1] sol[end][2]])
end
s = format("/N={:,d}_mu=[{:,.2f},{:,.2f}]_analytical.dat", N,u0[1],u0[2]) ;
CSV.write(data_path*s,df,writeheader=false)
Plots.plot(df.x_eq,df.y_eq,linestyle = :dot,m=(:hexagon, 2, 0.6),label="Equilibrium")
Plots.plot!((u0[1],u0[2]),m=(:hexagon),label="Starting")

sol.t


#test
p = [.05 1.] #beta, gamma so that p[1]/p[2] = R0
#Solve
prob = ODEProblem(sir, u0, tspan,p)
sol = solve(prob,Vern7(),abstol=1e-100,reltol=3e-14)

Plots.plot!(sol,vars=(1,2))
Plots.plot!((u0[1],u0[2]),m=(:hexagon),label="[x0,y0]")
