function [pop,cost,individual,feel]=popSort(pop,NP,cost,label,individual,feel)
% Ã°ÅÝÅÅÐò
for i=1:NP-1
        for j=1:NP-i
            if cost(j)>cost(j+1)
                tempcost=cost(j);
                cost(j)=cost(j+1);
                cost(j+1)=tempcost;
                
                temppop=pop(:,:,j);
                pop(:,:,j)=pop(:,:,j+1);
                pop(:,:,j+1)=temppop;
                
                templabel=label(j,:);
                label(j,:)=label(j+1,:);
                label(j+1,:)=templabel;
                
                tempindividual=individual(j);
                individual(j)=individual(j+1);
                individual(j+1)=tempindividual;
                
                tempfeel=feel(:,i);
                feel(:,i)=feel(:,j);
                feel(:,j)=tempfeel;
            end
        end
end
return;