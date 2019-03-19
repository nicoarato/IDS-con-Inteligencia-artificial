function [X, Y]= actualiza_datos(datos,yd,ydelete)
    [fila,~]= size(datos);
    X=[];
    Y=[];
    for i=1 : fila
    
        if(not(isequal(yd(i,:),ydelete)))
           X= [X; datos(i,:)]; 
           Y= [Y; yd(i,:)];
        end
    end


end