function [wcell]=inicializar_pesos(cant_c,cant_n,n)
    n=n+1; %Es para el sesgo
    wcell=cell(1,cant_c);                %Estructura pesos
    wcell{1,1}=-0.5+rand(cant_n(1),n);   %Inicializamos pesos de capa 1 segun la cantidad de entradas del archivo (patron)
   
    for c=2:cant_c
      wcell{1,c}=-0.5+rand(cant_n(c),cant_n(c-1)+1);  %Inicializamos los pesos de las demas capas con rand(cantidad de capas,sesgo+las salidas de la capa anterior) 
    end
end