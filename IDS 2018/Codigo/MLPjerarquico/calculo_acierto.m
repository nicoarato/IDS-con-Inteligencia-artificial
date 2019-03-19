%Calculo porcentaje acierto de perceptrón multicapa

function [t_acierto]=calculo_acierto(X,yd1, yd2, yd3,wcell,cant_c,alfa_sigm)
    m=length(yd1);
    c_acierto=0;
    ycell=cell(1,cant_c+1); %Estructura ycell={entradas=x | salidas de cada capa=yc}
    
    for p=1:m               %recorro cada patrón
        ycell{1,1}=X(p,:)'; %Vector columna con las entradas de un archivo (patron)

        for c=1:cant_c      %recorro capas
           Wc=wcell{1,c};   %Matriz de pesos de la capa c.

           if(c~=cant_c+1)           %Mientras no este en la ultima capa agrego sesgo
               Xc=[-1; ycell{1,c}]; 
           else
               Xc= ycell{1,c};
           end
           
           y=Wc*Xc; 
           y=sigmoidea(y,alfa_sigm); %Producto punto con función de activación aplicada    
           ycell{1,c+1}=y;
        end

        if ( (sign(y(1))==yd1(p)) && (sign(y(2))==yd2(p)) && (sign(y(3))==yd3(p)) )
           c_acierto=c_acierto+1; 
        end
    
    end
                       
        t_acierto=c_acierto/m*100; % Tasa de Aciertos
  end