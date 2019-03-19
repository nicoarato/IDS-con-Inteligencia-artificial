% MLP.

clc;
close all;
clear all;
 

max_epocas=100;
gamma=0.05;         %Tasa de aprendizaje
umbral_acierto=99;
cant_part=1;        %Cantidad de particiones del archivo
porc_trn=80;        %Porcentaje para entrenamiento 
alfa_sigmoidea=1; 
t_momento=0.4;      %Término momento

x_trn=csvread('..\Datos\train_datos_norm.csv');
yd_trn=csvread('..\Datos\etiquetas.csv'); 

validacion_cruzada([x_trn yd_trn],cant_part,porc_trn); %Particiona los datos para generar entrenamiento y prueba
MC=zeros(8,8);
for i=1 : cant_part
    
    x_trn=csvread(strcat('Particiones\trn','_',num2str(i),'.csv')); %Entrenamiento
    %yd_trn=csvread('..\Datos\etiquetas.csv'); %Entrenamiento
    yd_trn = x_trn(:,end-2:end);
    x_trn = x_trn(:,1:end-3);
    

    x_tst=csvread(strcat('Particiones\tst','_',num2str(i),'.csv')); %Prueba
    %yd_tst=csvread('..\Datos\test-etiquetas01.csv'); %Prueba
    yd_tst = x_tst(:,end-2:end);
    x_tst = x_tst(:,1:end-3);
    

    %Estructura de red
    cant_n=[39 20 3];           %cantidad neuronas por capas
%   cant_n=[39 20 10 5 3];
    cant_c=length(cant_n);  %cantidad capas

    [~,n]=size(x_trn);

    %Inicializamos pesos aleatorios para una red determinada
    [wcell]=inicializar_pesos(cant_c,cant_n,n); %n cantidad de entradas


    %ENTRENAMIENTO Y PRUEBA CON TERMINO DE MOMENTO
    display(strcat('RESULTADOS con particion_ ',num2str(i),':') ) ;
    [w]=entrenamiento_sm([x_trn yd_trn],cant_c,max_epocas,alfa_sigmoidea,gamma,umbral_acierto,wcell);
    [t_acierto, m_Conf]=prueba_multicapa(x_tst,yd_tst,w,cant_c,alfa_sigmoidea);
    MC=MC+m_Conf;
end
MC=MC./cant_part;
[mconf,recall,precision,miss_rate,accuracy]=medidas(MC);
