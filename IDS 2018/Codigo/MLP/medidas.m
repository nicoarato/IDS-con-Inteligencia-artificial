function [mconf,recall,precision,miss_rate,accuracy]=medidas(MC)
%         recall=0;      % TP/(TP+FN)
        %specificity=0; %TN/(TN+FP)
%         precision=0;   %TP/(TP+FP)
%         miss_rate=1-recall;    %FN/(FN+TP)
        %accuracy=0;     %TP+TN/(TP+TN+FP+FN)
        FP=0;
        TP=0;
        FN=0;
        [m,~]=size(MC);
%         TP serán los ataques , lo que interesa clasificar // personas
%         enfermas
        for i=2 : m
             TP = TP + MC(i,i);
        end
        
        
        for i=2:m
              
                FP=FP+ MC(1,i); %Normales clasificados como ataques
        end
      
        
        for i=2 : m
            FN=FN+MC(i,1); %Ataques clasificados como normales IMPORTANTE
        end
        
%         Normales serán los TN;
        TN=MC(1,1);
        
        recall= TP/(TP+FN);
        precision=TP/(TP+FP);
        miss_rate= 1-recall;
        accuracy=(TP+TN)/(TP+TN+FP+FN);
        
        mconf= [TP FP; FN TN];
        
end