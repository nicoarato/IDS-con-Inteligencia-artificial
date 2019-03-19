function [indice] = decodificacion(y)
    if (isequal(y, [-1 -1 -1]))
        indice = 1; %Clase normal
    end
    if (isequal(y, [-1 -1 1]))
        indice = 2; %Clase Neptune
    end
    if (isequal(y, [-1 1 -1]))
        indice = 3; %Clase smurf
    end
    if (isequal(y, [-1 1 1]))
        indice = 4; %Warezclient
    end
    if (isequal(y, [1 -1 -1]))
        indice = 5; %Clase ipsweep
    end
    if (isequal(y, [1 -1 1]))
         indice = 6; %Clase nmap
    end
    if (isequal(y, [1 1 -1]))
        indice = 7; %Clase portsweep
    end
    if (isequal(y, [1 1 1]))
        indice = 8; %satan
    end
    
    
end