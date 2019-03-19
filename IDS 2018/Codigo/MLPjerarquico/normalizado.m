X_train = csvread('KDDTrain+.csv');

m = X_train;

mins= zeros(1,42);
maxs= zeros(1,42);

[nfilas,ncols]= size(m);

for i=1: ncols
    
    mins(i) = min(m(:,i));
    maxs(i) = max(m(:,i));

end
%%
[nfilas,ncols]= size(X_train);

for j= 1 : ncols
    for i = 1 : nfilas
        X_train(i,j)=(X_train(i,j)-mins(j))/(maxs(j)-mins(j)); %actualizo X_train normalizado
    end
    
    
end

%%
%Elimino columna 8, 20 y 21 que vale siempre 0 y me hacen datos NAN al
%normalizar

X_train = [X_train(:,1:19) X_train(:, 22:end)];
X_train = [X_train(:,1:7) X_train(:, 9:end)];

%%
%Guardo datos

csvwrite('train_datos_norm.csv',X_train);




% csvwrite(strcat('D:\FACULTAD\inteligencia\drive inteligencia\TP2\Ejercicio_1\Particiones\tst_ej',num2str(nro_ejercicio),'_',num2str(i),'.csv'),x_tst);