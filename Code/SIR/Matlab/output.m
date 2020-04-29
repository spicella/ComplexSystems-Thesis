function r=output( t, x, status );

global t_end N XX YY NN;

 r=0;
if length( status )==0

    %disp([8 '.']);

    %%disp(t);
   %{0
 NN=NN+1;
    if NN>1000
        
        NN=0;
        
        r=1;
        
        return;
        
    end;
    %}
   
end;
