%Prueba para perceptrón multicapa

function [t_acierto]=prueba_multicapaMLP1(X,yd,wcell,cant_c,alfa_sigm)
    c_acierto=0;
    m_Conf_1=zeros(2,2);
    col=0;
    fila=0;
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

        if ( sign(y)==yd(p) )
           c_acierto=c_acierto+1;
           
        end
        if ( yd(p)== -1)
            fila=1;
        else 
            fila=2;
        end
        if (sign(y)==-1)
            col=1;
        else
            col=2;
        end
        
      m_Conf_1(fila,col)= m_Conf_1(fila,col)+1;   
    end
                       
    t_acierto=c_acierto/m*100; % Tasa de Aciertos
    display('Resultados Pruebas:');
    display(['     Porcentaje de aciertos: ',num2str(t_acierto)]);
    display('------------------------------------------');
    display('Matriz de confusión');
    m_Conf_1
end