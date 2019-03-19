%x= vector 
%a= parametro de la funcion

function [ys] = sigmoidea(x,a) 
    N=length(x);
    ys=zeros(N,1);  %vector columna con las salidas de la sigmoidea
    for i=1:N
       ys(i)=(1-exp(-a*x(i)))/(1+exp(-a*x(i))); 
    end 
end

