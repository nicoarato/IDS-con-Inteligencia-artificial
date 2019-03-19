%Entrenamiento para perceptrón multicapa

function [wcell]=entrenamiento_sm(x,cant_c,max_epocas,alfa_sigm,gamma,umbral_acierto,wcell)
 
    c_epocas=0;
    [m,n]=size(x);
     
    %Separamos ydeseada y las entradas del archivo
    yd1=x(:,n-2);
    yd2=x(:,n-1);
    yd3=x(:,n);
    
    
    X=x(:,1:end-3);
    
    %Las entradas de una capa son las salidas de la capa anterior o lo que
    %es lo mismo, las salidas de una capa son las entradas de la capa
    %siguiente.  
    
    ycell=cell(1,cant_c+1); %Estructura ycell={entradas=x | salidas de cada capa=yc}  
   
    dcell=cell(1,cant_c); %Deltas de error - Estructura delta={delta de cada capa yc} - por cada neurona de cada capa tenemos un delta de error
    

    while(c_epocas<max_epocas) 
        for p=1:m    %recorro cada patrón

            ycell{1,1}=X(p,:)';  %Vector columna con las entradas de un archivo (patron)

            
            %PROPAGACIÓN HACIA ADELANTE (primer ida) -->>> :)
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

            %PROPAGACIÓN HACIA ATRÁS (vuelta)<<<---- :|

            %Como la red tiene una sola salida, lo hacemos asi 
            e1=yd1(p)-y(1);
            e2=yd2(p)-y(2);
            e3=yd3(p)-y(3);
            
            dcell{1,cant_c}=[(e1*1/2*(1-y(1))*(1+y(1))); (e2*1/2*(1-y(2))*(1+y(2))); (e3*1/2*(1-y(3))*(1+y(3)))];  %Delta de error total de la red (ultima capa)
            

            for c=cant_c-1:-1:1         %recorro capas

                d_derecha=dcell{1,c+1}; %DELTAS: Vector columna con errores de cada neurona de la capa de la derecha ->
                Wc=wcell{1,c+1};        
                Wc=Wc(:,2:end);         %Pesos: quito la primer columna (pesos del sesgo) y traspongo
                Yc=ycell{1,c+1};        %Salida de la capa que le calculo delta

                dcell{1,c}=Wc'*d_derecha*1/2.*(1-Yc).*(1+Yc); %deltas por capa (capas ocultas)
            end

            
            %CORRECCIÓN DE PESOS (segunda ida)-->>> :(
            for c=1:cant_c              %recorro capa
                    dc=dcell{1,c};          %Deltas de error de las neuronas de la capa c (vector = cantidad neuronas de la capa)
                    Xc=[-1 ;ycell{1,c}];    %Entradas de la capa c (Vector columna)
                    dW=gamma*dc*Xc';        %Delta de pesos
                    wcell{1,c}=wcell{1,c}+dW; %Actualización pesos: w_nuevo=w_actual+delta de pesos
            end
 
        end %for  patron

       %Gráfico cada tantos números de épocas = 25
%         if (mod(c_epocas,50)==0)
%             graficar_ejercicio3(X,yd,wcell,cant_c,alfa_sigm);title('Entrenamiento SIN TÉRMINO DE MOMENTO');
%             figure(1);
%         end
        
        %TASA DE ACIERTO DE EPOCA
        [t_acierto]=calculo_acierto(X,yd1,yd2,yd3,wcell,cant_c,alfa_sigm);
        
        if (t_acierto>umbral_acierto)
                 display('Resultados entrenamiento:');
                 display(['     Porcentaje de Aciertos : ',num2str(t_acierto)]);
                 display(['     Cantidad de epocas: ',num2str(c_epocas)]);
                 %graficar_ejercicio3(X,[yd1 yd2 yd3],wcell,cant_c,alfa_sigm);title('Entrenamiento SIN TÉRMINO DE MOMENTO');
                 return;  
        end

        c_epocas=c_epocas+1;  
   
    end %while
    
    display('Resultados entrenamiento:');
    display(['     Se llego al numero maximo de epocas ',num2str(c_epocas)]);
    %graficar_ejercicio3(X,yd,wcell,cant_c,alfa_sigm);title('Entrenamiento SIN TÉRMINO DE MOMENTO');

end %function