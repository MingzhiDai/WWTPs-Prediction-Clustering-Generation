function [cost,center,label2]= getFinalCost(data,label,k)
cost=0;
[r,c]=size(data);
count=zeros(1,k);
% 初始化center count
center_temp=zeros(k,c);

% 重计算 center count
for i=1:r
    for j=1:k
        if label(i)==j
            center_temp(j,:)= center_temp(j,:)+data(i,:);
            count(j)=count(j)+1;
        end
    end
end

center2=zeros(k,c);
for i=1:k
    if count(i)>0
         center2(i,:)=  center_temp(i,:)/count(i);
    end  
end

[r3,c3]=size(find(count>1));
center=zeros(c3,c);
biaoji=1;
% 得到最终center
for i=1:k
    if count(i)>1
         center(biaoji,:) = center_temp(i,:)/count(i);
         biaoji=biaoji+1;
    end  
end

[cost,label2]=caculateCost(center2,label,data,k);

for i=1:k
   [hang,lie]=size(find(label2==i));
   number(i)=lie;
end
% number

return;
