%DEVUELVE LA PREDICCION DE UN MLP

% function [t_acierto,y, m_Conf]=prueba_multicapa_jerarquico(X,yd,wcell,cant_c,alfa_sigm)

% Le paso un patron y pruebo si pertenece o no a tipo YD
function [yy]=prueba_multicapa_jerarquico(X,wcell,cant_c,alfa_sigm)

    
    [m,~]=size(X);


    ycell=cell(1,cant_c+1); %Estructura ycell={entradas=x | salidas de cada capa=yc}
    
    for p=1:m    %recorro cada patrón
        ycell{1,1}=X(p,:)';  %Vector columna con las entradas de un archivo (patron)

        for c=1:cant_c                %recorro capas
           Wc=wcell{1,c};            %Matriz de pesos de la capa c.

           if(c~=cant_c+1)           %Mientras no este en la ultima capa agrego sesgo
               Xc=[-1; ycell{1,c}]; 
           else
               Xc= ycell{1,c};
           end
           
           y=Wc*Xc; 
           y=sigmoidea(y,alfa_sigm); %Producto punto con función de activación aplicada    
           ycell{1,c+1}=y;
        end
        
    end
    yy= sign(y);
%        yy= sigmoidea(y,alfa_sigm) 

end