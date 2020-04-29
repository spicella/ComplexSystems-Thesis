%%
function []=sir_cme(N,m1,m2,s1,s2,beta)
%%
tic
%m1,m2 == mean on each axis assuming gaussian profile
%s1,s2 == std dev of gaussian on each axis
%% Constants
    
    global t_end N XX YY NN;
    t_end = 1e2;
    gamma = 1

%% Initial conditions

DoF = N^2;

[XX YY] = meshgrid( 0:N-1 );

X=XX(1:end)';
Y=YY(1:end)';


Z        = 1/( 2 * pi * sqrt(s1 * s2)  )* ...
          exp( -( X - m1 ).^2 / 2 / s1 ).* ...
          exp( -( Y - m2 ).^2 / 2 / s2 );
      
Z=sparse(Z);
%% Exact operators

E_1   = sparse( diag( ones( N-1, 1 ), -1 ));
Em1   = sparse( diag( ones( N-1, 1 ),  1 ));

T01 = kron( eye(N), E_1 );
T10 = kron( E_1, eye(N) );

T0m1 = kron( eye(N), Em1 );
Tm10 = kron( Em1, eye(N) );

TWx1   = sparse( 1:N^2, 1:N^2, X );
TWx2   = sparse( 1:N^2, 1:N^2, Y );
I  =  sparse( 1:N^2, 1:N^2, 1 )
XY = TWx1*TWx2


%v0     
%J =     (beta/N) * (Tm10 *XY - XY) + ...    %reaction1: decrease of S   
%        (beta/N) * (T01  *XY - XY) +  ...   %reaction2: increase of I
%         gamma * (T0m1 * TWx2 - TWx2)       %reaction3: decrease of I

%v1     
%J =     (beta/N) * (T01  *XY - XY) +  ...   %reaction2: increase of I
%         gamma * (T0m1 * TWx2 - TWx2)       %reaction3: decrease of I

%v2
J = beta/N  * ( Tm10*T01  - I)* TWx2 * TWx1 +...    %(S,I)->(S-1,I+1)
    gamma   * ( T0m1  -   I)* TWx2;                 %(S,I)->(S,I-1)
 
 
clear E_1 Em1 T01 T10 Tm10 T0m1 TWx1 TWx2 I;           

%% Time integration
time=0;
NN=0;
xlim([0 N]);
ylim([0 N]);
frame=0;
All={};
disp('Time integration ');
R=Z';

opt.OutputFcn   = @output;
opt.Jacobian    = @(t,F) J;
opt.AbsTol      = 1e-100;
opt.stats       = 1; 
opt.RelTol      = 3e-14;

while time<t_end
    
    OutputAndPlot;
    

    [T R]=ode113( @(t,x) J*x, [time t_end], Z, opt );
    
    opt.InitialStep = T( end ) - T( end-1 );
    time = T(end);
    Z  = R( end, : )';
    
end;
OutputAndPlot;
disp('[ OK ]'); 

output_mat = reshape(Z,[N,N])

folder_name = sprintf('data_sir_v1/%d_%d_%d_%d_%d/CME/',N,m1,m2,s1,s2)
mkdir(folder_name)
filename = sprintf('data_sir_v1/%d_%d_%d_%d_%d/CME/%.3e.csv',N,m1,m2,s1,s2,beta/gamma)
dlmwrite(filename,full(output_mat));
toc
clearvars -global

end