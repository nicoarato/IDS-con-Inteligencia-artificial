%Prueba para perceptr�n multicapa

function [t_acierto, m_Conf]=prueba_multicapa(X,yd,wcell,cant_c,alfa_sigm)
    c_acierto=0;
    m_Conf=zeros(8,8);
    
    [m,~]=size(X);


    ycell=cell(1,cant_c+1); %Estructura ycell={entradas=x | salidas de cada capa=yc}
    
    for p=1:m    %recorro cada patr�n
        ycell{1,1}=X(p,:)';  %Vector columna con las entradas de un archivo (patron)

        for c=1:cant_c                %recorro capas
           Wc=wcell{1,c};            %Matriz de pesos de la capa c.

           if(c~=cant_c+1)           %Mientras no este en la ultima capa agrego sesgo
               Xc=[-1; ycell{1,c}]; 
           else
               Xc= ycell{1,c};
           end
           
           y=Wc*Xc; 
           y=sigmoidea(y,alfa_sigm); %Producto punto con funci�n de activaci�n aplicada    
           ycell{1,c+1}=y;
        end

        if ((sign(y(1))==yd(p,1)) && (sign(y(2))==yd(p,2)) && (sign(y(3))==yd(p,3)))
           c_acierto=c_acierto+1; 
        end
        
        fila = decodificacion(yd(p,:));
        columna = decodificacion(sign(y'));
        m_Conf(fila, columna) = m_Conf(fila, columna) + 1;
        
    
    end
                       
    t_acierto=c_acierto/m*100; % Tasa de Aciertos
    display('Resultados Pruebas:');
    display(['     Porcentaje de aciertos: ',num2str(t_acierto)]);
end