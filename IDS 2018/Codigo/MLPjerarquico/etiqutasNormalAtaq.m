clc;
close all;
clear

yd=csvread('..\Datos\etiquetas.csv');

[m, n] = size(yd);
yd_NormAtaq = zeros(m,1);

% -1 si es Normal, 1 si es Ataque
for i=1 : m
   if(yd(i,1) == -1 && yd(i,2) == -1 && yd(i,3) == -1)
       yd_NormAtaq(i) = -1;
   else
       yd_NormAtaq(i) = 1;
   end
end

csvwrite('..\Datos\etiquetasNormAtaq.csv',yd_NormAtaq);

