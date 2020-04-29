function []=sis_ode(N,S0,I0,beta)
%S0, I0 are fraction of total population, gamma = 1  
%N in argument only for name in save
t_end = 1e2;
gamma = 1;
% x(1),x(2) == S, I
    function dxdt = sir_ode(t,x) 
        dxdt = [gamma*x(2)-beta*x(1)*x(2);beta*x(1)*x(2)-gamma*x(2)];
    end
opts = odeset('Reltol',3e-14,'AbsTol',1e-100,'Stats','on');
tspan = [0 t_end];
bc = [S0 I0]; % BoundaryConditions
[t,y] = ode113(@sir_ode, tspan, bc, opts);

%legend('S','I','R','Location','SouthEast')
%subplot(1,2,1)
%str = sprintf('SIS model ODE solution, S(t), I(t), R(t)\n(S0,I0,beta) = (%.3f,%.3f,%.3f)',bc(1), bc(2), beta)
%plot(t,y(:,1))
%hold on 
%plot(t,y(:,2))

%title(['SIS ODE solution, R0 = ' num2str(beta/gamma)])
%xlabel('Time');
%ylabel('% of population');
%legend('S','I')
%ylim([0 1])
%hold on


plot(y(:,1),y(:,2))
title(['I(S(t)), SIS ODE solution, R0 = ' num2str(beta/gamma)])
xlabel('S');
ylabel('I');
legend('I(S)')
xlim([0 1])
ylim([0 1])
hold off


folder_name = sprintf('data_sis_v0/%d_%d_%d_%d_%d/ODE/',N,int32(N*S0),int32(N*I0),int32(N*.1),int32(N*.1))
mkdir(folder_name)
filename = sprintf('data_sis_v0/%d_%d_%d_%d_%d/ODE/%.3e.csv',N,int32(N*S0),int32(N*I0),int32(N*.1),int32(N*.1),beta/gamma)
output = [t,y(:,1), y(:,2)]; 
dlmwrite(filename,full(output));
clearvars -global

end

