 function [distance]=Distance(LVQ,data)
distance=0;
[row,~]=size(data);
for i=1:row
    distance=distance+pdist2(LVQ,data(i,:));
end
% clc;
% clear;
% a=[1 2 3 4 5 6 7 8 9 10];
% b=a(randperm(length(a),1));
% % i=0;
% a=[1, 5 ,7];
% % [~,i]=min(a);
% % i
% i=find(a==7);
% i;
% [Value,row]=min(sqrt(sum((C-repmat(p,size(C(:,1)))).^2,2)))
% return;
% Value=0;
% % row=zeros(1,4);
% C=[[1,1,1,1];
%     [2,2,2,2]];
% p=[1,1,1,1];
% q=[2,2,1,2];
% distance=pdist2(p,q);
% [Value,row]=min(sqrt(sum((C-repmat(p,size(C(:,1)))).^2,2)));
% [Value,rowtemp]=sort(p,'descend');
% labeltemps=rowtemp(1:3);
% for s=1:k
%     [Value,labeltemp]=min(sqrt(sum((data(label(cluster))-repmat(elit(labeltemp(s),:,i),size(data(label(cluster(:,1)))).^2,2)))));
%     datalabel(find(datalabel==0))=labeltemps(labeltemp)
    
        
