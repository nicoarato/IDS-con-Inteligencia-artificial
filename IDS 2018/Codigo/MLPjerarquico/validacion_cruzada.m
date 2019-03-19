function validacion_cruzada(x,cant_par,porc_trn)
    display('Validacion Cruzada....');
    porc_tst=100-porc_trn; %Porcentaje de datos de entrenamiento
    [m,n]=size(x);
    
    cant_tst=round(m*porc_tst/100);  %Cantidad de datos de prueba
    cant_trn=m-cant_tst;       %Cantidad de datos de entrenamiento
  
    x_tst=zeros(cant_tst,n);
    x_trn=zeros(cant_trn,n);
    for i=1:cant_par   %cantidad particiones
         ind=randperm(m);
         
         for j=1:cant_tst               %Datos de prueba  
            x_tst(j,:)=x(ind(j),:);
         end
          
         csvwrite(strcat('Particiones\tst','_',num2str(i),'.csv'),x_tst);
         
         for k=1:cant_trn                %Datos de Entrenamiento
            j=j+1;
             x_trn(k,:)=x(ind(j),:);
           
         end
          csvwrite(strcat('Particiones\trn','_',num2str(i),'.csv'),x_trn);
    end
    
end