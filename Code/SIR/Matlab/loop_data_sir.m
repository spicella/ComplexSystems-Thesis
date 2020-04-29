N=200
S0 = .9;
I0 = .1;
for i=0:20
    
    new = (0+i*.2);
    disp(new); 
        sir_ode(N,S0,I0,new)
        sir_cme(N,N*S0,N*I0,N*.1,N*.1,new)
end
S0 = .5;
I0 = .5;
for i=0:20
    new = (0+i*.2);
    disp(new); 
        sir_ode(N,S0,I0,new)
        sir_cme(N,N*S0,N*I0,N*.1,N*.1,new)
end
S0 = .1;
I0 = .9;
for i=0:20
    new = (0+i*.2);
    disp(new); 
        sir_ode(N,S0,I0,new)
        sir_cme(N,N*S0,N*I0,N*.1,N*.1,new)
end

