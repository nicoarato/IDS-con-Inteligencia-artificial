% MLP to mlp Jerarquico.
%Para cada partición, se actualizan los datos de entrenamiento en cada
%nivel (se utiliza actualiza_datos() para eliminar etiquetas)


clc;
close all;
clear all;

% Cargo de datos
x=csvread('C:\Users\nico\Desktop\TP final 15-11-2018\Datos\train_datos_norm.csv');
yd=csvread('C:\Users\nico\Desktop\TP final 15-11-2018\Datos\etiquetas.csv'); 
yd_NormAtaq=csvread('C:\Users\nico\Desktop\TP final 15-11-2018\Datos\etiquetasNormAtaq.csv'); % los guarda en la ultima columna de la particion

cant_part=5;        %Cantidad de particiones del archivo
porc_trn=80;        %Porcentaje para entrenamiento

validacion_cruzada([x yd],cant_part,porc_trn); %Particiona los datos para generar entrenamiento y prueba

%% Tipos de ataques para ir actualizando la X_TRN para cada MLP
normal=     [-1 -1 -1];
neptune=    [-1 -1 1];
smurf=      [-1 1 -1];
warezclient=[-1 1 1];
ipsweep=    [1 -1 -1];
nmap=       [1 -1 1];
portsweep=  [1 1 -1];
satan=      [1 1 1];


%% MATRIZ DE CONFUSION GLOBAL
MC=zeros(8,8);

%%

for i=1 : cant_part
    %% Primer MLP (Separa normales de ataques) NORMAL
    
    max_epocas1=100;
    gamma1=0.05;         %Tasa de aprendizaje
    umbral_acierto1=99; 
    alfa_sigmoidea1=1; 
   % t_momento=0.4;      %Término momento
    %Estructura de red
    cant_n1=[39 10 5 3];           %cantidad neuronas por capas
    cant_c1=length(cant_n1);  %cantidad capas
    
    x_trn=csvread(strcat('Particiones\trn','_',num2str(i),'.csv')); %Entrenamiento
    yd_trn = x_trn(:,end-2:end);
    x_trn = x_trn(:,1:end-3);
    

    x_tst=csvread(strcat('Particiones\tst','_',num2str(i),'.csv')); %Prueba
    yd_tst = x_tst(:,end-2:end);
    x_tst = x_tst(:,1:end-3);
    

    [~,n]=size(x_trn);

    %Inicializamos pesos aleatorios para una red determinada
    [wcell1]=inicializar_pesos(cant_c1,cant_n1,n); %n cantidad de entradas


    %ENTRENAMIENTO Y PRUEBA 
    display(strcat('RESULTADOS Normales-Ataques particion_ ',num2str(i),':') ) ;
    display('Normal');
    [w1]=entrenamiento_sm([x_trn yd_trn],cant_c1,max_epocas1,alfa_sigmoidea1,gamma1,umbral_acierto1,wcell1,normal);
    
%     [t_acierto, m_Conf]=prueba_multicapa(x_tst,yd_tst,w1,cant_c,alfa_sigmoidea); % calcula acierto y matriz de confusión.
%     m_Conf
    % tengo que sacar los datos NORMALES

      [x_trn, yd_trn]=actualiza_datos(x_trn,yd_trn,normal);  %% yd_train y x_trn para la siguiente MLP
    
      
    %% Segundo MLP NEPTUNE
    
    max_epocas2=100;
    gamma2=0.05;         %Tasa de aprendizaje
    umbral_acierto2=99; 
    alfa_sigmoidea2=1; 

    %Estructura de red
    cant_n2=[39 8 4 3];           %cantidad neuronas por capas
    cant_c2=length(cant_n2);  %cantidad capas
    
    %Inicializamos pesos aleatorios para una red determinada
    [wcell2]=inicializar_pesos(cant_c2,cant_n2,n); %n cantidad de entradas
    
    %ENTRENAMIENTO Y PRUEBA
%     display(strcat('RESULTADOS Ataques particion_ ',num2str(i),':') ) ;
    display('neptune');
    [w2]=entrenamiento_sm([x_trn yd_trn],cant_c2,max_epocas2,alfa_sigmoidea2,gamma2,umbral_acierto2,wcell2,neptune);
   
%     [t_acierto, m_Conf]=prueba_multicapa(x_tst,yd_tst,w2,cant_c,alfa_sigmoidea);
%     m_Conf
    
    
    [x_trn, yd_trn]=actualiza_datos(x_trn,yd_trn,neptune);  %% yd_train y x_trn para la siguiente MLP
    
   %% Tercer MLP SMURF
    
    max_epocas3=100;
    gamma3=0.05;         %Tasa de aprendizaje
    umbral_acierto3=99; 
    alfa_sigmoidea3=1; 
    %Estructura de red
    cant_n3=[39 20 10 3];           %cantidad neuronas por capas
    cant_c3=length(cant_n3);  %cantidad capas
    
    
    %Inicializamos pesos aleatorios para una red determinada
    [wcell3]=inicializar_pesos(cant_c3,cant_n3,n); %n cantidad de entradas
    
    %ENTRENAMIENTO Y PRUEBA 
%     display(strcat('RESULTADOS Ataques particion_ ',num2str(i),':') ) ;
    display('smurf');  
    [w3]=entrenamiento_sm([x_trn yd_trn],cant_c3,max_epocas3,alfa_sigmoidea3,gamma3,umbral_acierto3,wcell3,smurf);
    
%     [t_acierto, m_Conf]=prueba_multicapa(x_tst,yd_tst,w3,cant_c,alfa_sigmoidea);
%     m_Conf    
    
     
    [x_trn, yd_trn]=actualiza_datos(x_trn,yd_trn,smurf); %% yd_train y x_trn para la siguiente MLP
 %% Cuarto MLP warezclient
    
    max_epocas4=100;
    gamma4=0.05;         %Tasa de aprendizaje
    umbral_acierto4=99; 
    alfa_sigmoidea4=1; 
    %Estructura de red
    cant_n4=[39 15 5 3];           %cantidad neuronas por capas
    cant_c4=length(cant_n4);  %cantidad capas
   
    
    %Inicializamos pesos aleatorios para una red determinada
    [wcell4]=inicializar_pesos(cant_c4,cant_n4,n); %n cantidad de entradas
    
    %ENTRENAMIENTO Y PRUEBA
%     display(strcat('RESULTADOS Ataques particion_ ',num2str(i),':') ) ;
    display('warezclient');
    [w4]=entrenamiento_sm([x_trn yd_trn],cant_c4,max_epocas4,alfa_sigmoidea4,gamma4,umbral_acierto4,wcell4,warezclient);
%     [t_acierto, m_Conf]=prueba_multicapa(x_tst,yd_tst,w4,cant_c,alfa_sigmoidea);
%     m_Conf        
    
    
    [x_trn, yd_trn]=actualiza_datos(x_trn,yd_trn,warezclient); %% yd_train y x_trn para la siguiente MLP
    
    %% QUINTO MLP ipsweep
    
    max_epocas5=100;
    gamma5=0.05;         %Tasa de aprendizaje
    umbral_acierto5=99; 
    alfa_sigmoidea5=1; 
    %Estructura de red
    cant_n5=[39 25 10 3];           %cantidad neuronas por capas
    cant_c5=length(cant_n5);  %cantidad capas
    
    %Inicializamos pesos aleatorios para una red determinada
    [wcell5]=inicializar_pesos(cant_c5,cant_n5,n); %n cantidad de entradas
    
    %ENTRENAMIENTO Y PRUEBA
%     display(strcat('RESULTADOS Ataques particion_ ',num2str(i),':') ) ;
    display('ipsweep');
    [w5]=entrenamiento_sm([x_trn yd_trn],cant_c5,max_epocas5,alfa_sigmoidea5,gamma5,umbral_acierto5,wcell5,ipsweep);
%     [t_acierto, m_Conf]=prueba_multicapa(x_tst,yd_tst,w5,cant_c,alfa_sigmoidea);
%     m_Conf        
    
    
    [x_trn, yd_trn]=actualiza_datos(x_trn,yd_trn,ipsweep); %% yd_train y x_trn para la siguiente MLP
    
     %% SEXTO MLP nmap
    
    max_epocas6=100;
    gamma6=0.05;         %Tasa de aprendizaje
    umbral_acierto6=99; 
    alfa_sigmoidea6=1;
    %Estructura de red
    cant_n6=[39 30 20 10 3];           %cantidad neuronas por capas
    cant_c6=length(cant_n6);  %cantidad capas
    
    %Inicializamos pesos aleatorios para una red determinada
    [wcell6]=inicializar_pesos(cant_c6,cant_n6,n); %n cantidad de entradas
    
    %ENTRENAMIENTO Y PRUEBA
%     display(strcat('RESULTADOS Ataques particion_ ',num2str(i),':') ) ;
    display('nmap');
    [w6]=entrenamiento_sm([x_trn yd_trn],cant_c6,max_epocas6,alfa_sigmoidea6,gamma6,umbral_acierto6,wcell6,nmap);
%     [t_acierto, m_Conf]=prueba_multicapa(x_tst,yd_tst,w6,cant_c,alfa_sigmoidea);
%     m_Conf        
    
    
    [x_trn, yd_trn]=actualiza_datos(x_trn,yd_trn,nmap); %% yd_train y x_trn para la siguiente MLP
    
      %% SEPTIMO MLP portsweep y satan
    
    max_epocas7=100;
    gamma7=0.05;         %Tasa de aprendizaje
    umbral_acierto7=99; 
    alfa_sigmoidea7=1;
    %Estructura de red
    cant_n7=[39 17 9 3];           %cantidad neuronas por capas
    cant_c7=length(cant_n7);  %cantidad capas
    
    %Inicializamos pesos aleatorios para una red determinada
    [wcell7]=inicializar_pesos(cant_c7,cant_n7,n); %n cantidad de entradas
    
    %ENTRENAMIENTO Y PRUEBA
%     display(strcat('RESULTADOS Ataques particion_ ',num2str(i),':') ) ;
    display('portsweep y satan');
    [w7]=entrenamiento_sm([x_trn yd_trn],cant_c7,max_epocas7,alfa_sigmoidea7,gamma7,umbral_acierto7,wcell7,portsweep);
%     [t_acierto, m_Conf]=prueba_multicapa(x_tst,yd_tst,w7,cant_c,alfa_sigmoidea);
%     m_Conf        
    
    display('----------------------------------------');
   
%% Prueba sobre todos los MLPs


% [t_acierto, m_Conf]=prueba_multicapa_global(x_tst,yd_tst,w2,cant_c,alfa_sigmoidea);
%  m_Conf
[t_acierto, m_Conf]=prueba_multicapa_global(x_tst,yd_tst,w1,cant_c1,alfa_sigmoidea1,w2,cant_c2,alfa_sigmoidea2,w3,cant_c3,alfa_sigmoidea3,w4,cant_c4,alfa_sigmoidea4,w5,cant_c5,alfa_sigmoidea5,w6,cant_c6,alfa_sigmoidea6,w7,cant_c7,alfa_sigmoidea7);

 m_Conf
MC=MC+m_Conf;

    
end
MC=MC./cant_part;
[mconf,recall,precision,miss_rate,accuracy]=medidas(MC);

