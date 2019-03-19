%Prueba para perceptrón multicapa

function [t_acierto, m_Conf]=prueba_multicapa_global(X,yd,wcell1,cant_c1,alfa_sigm1,wcell2,cant_c2,alfa_sigm2,wcell3,cant_c3,alfa_sigm3,wcell4,cant_c4,alfa_sigm4,wcell5,cant_c5,alfa_sigm5,wcell6,cant_c6,alfa_sigm6,wcell7,cant_c7,alfa_sigm7)
    c_acierto=0;
    m_Conf=zeros(8,8);
    [M,~]= size(X);
    
    
    %% Tipos de ataques para ir actualizando la X_TRN para cada MLP
normal=     [-1 -1 -1];
neptune=    [-1 -1 1];
smurf=      [-1 1 -1];
warezclient=[-1 1 1];
ipsweep=    [1 -1 -1];
nmap=       [1 -1 1];
portsweep=  [1 1 -1];
satan=      [1 1 1];

%%
    %%se cuenta cada acierto correcto, pues cada MLP está especializado en
    %%un ataque.
   for p=1 : M %Para cada patron
     
       [y]=prueba_multicapa_jerarquico(X(p,:),wcell1,cant_c1,alfa_sigm1); % PRUEBO el PATRON CON MLP1
       
       if(isequal(y',normal))
             
             fila = decodificacion(yd(p,:));       %fila de matriz de conf
             columna = decodificacion(sign(y'));   %columna de matriz de conf
             m_Conf(fila, columna) = m_Conf(fila, columna) + 1; % actualiza matriz
             if (fila==columna) 
                 c_acierto= c_acierto+1;         %calculo del acierto
             end
       else
           [y]=prueba_multicapa_jerarquico(X(p,:),wcell2,cant_c2,alfa_sigm2); % PRUEBO el PATRON CON MLP2
           if(isequal(y',neptune))
%              c_acierto= c_acierto+1;         %calculo del acierto
             fila = decodificacion(yd(p,:));       %fila de matriz de conf
             columna = decodificacion(sign(y'));   %columna de matriz de conf
             m_Conf(fila, columna) = m_Conf(fila, columna) + 1; % actualiza matriz
             if (fila==columna) 
                 c_acierto= c_acierto+1;         %calculo del acierto
             end   
           
           else
               [y]=prueba_multicapa_jerarquico(X(p,:),wcell3,cant_c3,alfa_sigm3); % PRUEBO el PATRON CON MLP3
               if(isequal(y',smurf))
%                  c_acierto= c_acierto+1;         %calculo del acierto
                 fila = decodificacion(yd(p,:));       %fila de matriz de conf
                 columna = decodificacion(sign(y'));   %columna de matriz de conf
                 m_Conf(fila, columna) = m_Conf(fila, columna) + 1; % actualiza matriz
                 if (fila==columna) 
                    c_acierto= c_acierto+1;         %calculo del acierto
                 end
               else
                   [y]=prueba_multicapa_jerarquico(X(p,:),wcell4,cant_c4,alfa_sigm4); % PRUEBO el PATRON CON MLP4
                   if(isequal(y',warezclient))
%                      c_acierto= c_acierto+1;         %calculo del acierto
                     fila = decodificacion(yd(p,:));       %fila de matriz de conf
                     columna = decodificacion(sign(y'));   %columna de matriz de conf
                     m_Conf(fila, columna) = m_Conf(fila, columna) + 1; % actualiza matriz
                    if (fila==columna) 
                        c_acierto= c_acierto+1;         %calculo del acierto
                    end
                   else
                       [y]=prueba_multicapa_jerarquico(X(p,:),wcell5,cant_c5,alfa_sigm5); % PRUEBO el PATRON CON MLP5
                       if(isequal(y',ipsweep))
%                          c_acierto= c_acierto+1;         %calculo del acierto
                         fila = decodificacion(yd(p,:));       %fila de matriz de conf
                         columna = decodificacion(sign(y'));   %columna de matriz de conf
                         m_Conf(fila, columna) = m_Conf(fila, columna) + 1; % actualiza matriz
                            if (fila==columna) 
                             c_acierto= c_acierto+1;         %calculo del acierto
                            end
                       else
                           [y]=prueba_multicapa_jerarquico(X(p,:),wcell6,cant_c6,alfa_sigm6); % PRUEBO el PATRON CON MLP6
                           if(isequal(y',nmap))
%                              c_acierto= c_acierto+1;         %calculo del acierto
                             fila = decodificacion(yd(p,:));       %fila de matriz de conf
                             columna = decodificacion(sign(y'));   %columna de matriz de conf
                             m_Conf(fila, columna) = m_Conf(fila, columna) + 1; % actualiza matriz
                            if (fila==columna) 
                                c_acierto= c_acierto+1;         %calculo del acierto
                            end
                           
                           else
                               [y]=prueba_multicapa_jerarquico(X(p,:),wcell7,cant_c7,alfa_sigm7); % PRUEBO el PATRON CON MLP7
                               if(isequal(y',portsweep))
%                                  c_acierto= c_acierto+1;         %calculo del acierto
                                 fila = decodificacion(yd(p,:));       %fila de matriz de conf
                                 columna = decodificacion(sign(y'));   %columna de matriz de conf
                                 m_Conf(fila, columna) = m_Conf(fila, columna) + 1; % actualiza matriz
                                    if (fila==columna) 
                                     c_acierto= c_acierto+1;         %calculo del acierto
                                    end
                               else
%                                  c_acierto= c_acierto+1;         %calculo del acierto
                                 fila = decodificacion(yd(p,:));       %fila de matriz de conf
                                 columna = decodificacion(sign(satan));   %columna de matriz de conf
                                 m_Conf(fila, columna) = m_Conf(fila, columna) + 1;
                                if (fila==columna) 
                                     c_acierto= c_acierto+1;         %calculo del acierto
                                end
                               end
                           end
                       end
                   end
               end
           end
       end
       
   end
                       
    t_acierto=c_acierto/M*100; % Tasa de Aciertos
    display('Resultados Pruebas:');
    display(['     Porcentaje de aciertos: ',num2str(t_acierto)]);
    display('------------------------------------------');
end