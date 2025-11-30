function [pop,Antenna_L,Antenna_R,pmu,pmodify,d_up,d_down,elit,individual]=initial(NP,k,data,elitnum)
[r,c]=size(data);
clear elit;
for i=1:elitnum
    elit(:,:,i)=zeros(k,c);
end
d_up=max(data);
d_down=min(data);
% d_down
% d_up
for i=1:NP
    individual(i)=i;
    for j=1:k
        for m=1:c
            pop(j,m,i)=rand*abs(d_up(m)-d_down(m))+d_down(m);
        end
    end
end
   
for i=1:NP
    for j=1:k
        for m=1:c
            Antenna_L(j,m,i)=rand*abs(d_up(m)-d_down(m))+d_down(m);
        end
    end
end

for i=1:NP
    for j=1:k
        for m=1:c
            Antenna_R(j,m,i)=rand*abs(d_up(m)-d_down(m))+d_down(m);
        end
    end
    
pmu(i)=0.05;
pmodify(i)=1;
end
% pop
% pmu
% pmodify
% cost
% d_up
% d_down
return;