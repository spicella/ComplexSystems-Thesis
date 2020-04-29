    disp(time);
    cla
    Z  = R( end, : )';
   % Z=v;
    ZZ = reshape( full( Z ), N, N );

    max_value = max(ZZ(1:end));
    % colormap hot;
    v = logspace( log10( max_value )-15, log10( max_value*0.99 ), 15); 
    contour( XX, YY, ZZ, v, 'color','k')
     
    view(0,90)
    %axis equal;
    title([' T = ' num2str(time,3)])
    refresh;
    drawnow;
    disp( max_value );
    frame=frame+1;
    axis square 
    All{end+1}.Z=Z;
    All{end}.T=time;
    set(gca,'FontName','Geneva')
    set(gca,'FontSize',14);
    
