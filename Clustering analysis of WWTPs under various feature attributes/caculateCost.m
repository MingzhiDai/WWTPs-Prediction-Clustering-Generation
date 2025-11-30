function [cost,label]=caculateCost(pop,label,iris,k)

[r,c]=size(iris);
count=zeros(1,k);
cost=0;
realc=zeros(k,c);
for p=1:r
    for q=1:k
        if label(p)==q 
            realc(q,:)=realc(q,:)+iris(p,:);%每种聚类所有data值相加（留作求质心，新的center）
            count(q)=count(q)+1;     %根据标签求出每种聚类的点数，
            cost=cost+norm(pop(q,:)-iris(p,:));  %求适应度值的分子
            break;
        end
    end %end col
end % end roll

% 得到类内样本数量为1的样本位置
count1=find(count==1);
[r1 c1]=size(count1);
weizhi=zeros(1,c1);
jishu=1;
for i=1:r
    for j=1:c1
        if label(i)==count1(j)
            weizhi(jishu)=i;
            jishu=jishu+1;
        end
    end
end

% 计算临时中心
count2=find(count>1);
[r2 c2]=size(count2);
center=zeros(c2,c);
jishu=1;
for i=1:k
    if count(i)>1
        center(jishu,:)=realc(i,:)/count(i);  %求质心，新的center
        duiying(jishu)=i;
        jishu=jishu+1;
        
    end
end

% 重新分配样本
[r3 c3]=size(weizhi);
for i=1:c3%样本
    labeltemp=1;
    for j=1:c2%类中心
        temp=norm(iris(weizhi(i),:)-center(j,:))-norm(iris(weizhi(i),:)-center(labeltemp,:));
        if temp<0;
            labeltemp=j;
        end
    end
    center(labeltemp,:)=(center(labeltemp,:)*count(labeltemp)+iris(labeltemp))/(count(labeltemp)+1);
    label(weizhi(i))=duiying(labeltemp);
end

% cost=cost/sum(pdist(center(:,:)))*c2*(c2-1)/2;
[r4 c4]=size(center);
if r4>1
    cost=cost/sum(pdist(center(:,:)))*r4*(r4-1)/2;   %求适应度值 %pdist：求center各对行向量的相互距离
    
else
    cost=inf;
end
return;