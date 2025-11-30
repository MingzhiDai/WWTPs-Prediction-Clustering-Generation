% clc;
% clear all;
function [elit]=EM_GMM(elit,original_data,clusters)
% filename='C:\Users\Dai MZ\Downloads\148个整理好的UCI数据集及相应代码\148个整理好的UCI数据集及相应代码\iris\预处理完成\iris.mat';
% Data=load(filename);
% all_data=Data.DataSet;
% original_data=all_data(:,2:5);
data=original_data';
[dim,Num]=size(data);
max_iter=10;%最大迭代次数
min_improve=1e-4;% 提升的精度
Ngauss=clusters;%混合高斯函数个数
Pw=zeros(1,Ngauss);%保存权重
mu=zeros(dim,Ngauss);%保存每个高斯分类的均值,每一列为一个高斯分量
mu_transpose=zeros(Ngauss,dim);
sigma= zeros(dim,dim,Ngauss);%保存高斯分类的协方差矩阵
fprintf('采用K均值算法对各个高斯分量进行初始化\n');
[cost,cm,cv,cc,cs,map] = vq_flat(data, Ngauss);%聚类过程  map:样本所对应的聚类中心
mu=cm;%均值初始化
for j=1:Ngauss
   gauss_labels=find(map==j);%找出每个类对应的标签
   Pw(j)= length(gauss_labels)/length(map);%类别为1的样本个数占总样本的个数 
   sigma(:,:,j)  = diag(std(data(:,gauss_labels),0,2)); %求行向量的方差，只取对角线，其他特征独立，并将其赋值给对角线
end

last_loglik = -Inf;%上次的概率
% 采用EM算法估计GMM的各个参数
if Ngauss==1  %一个高斯函数不需要用EM进行估计
    sigma(:,:,1)  = sqrtm(cov(data',1));
    mu(:,1)       = mean(data,2);
else
     sigma_i  = squeeze(sigma(:,:,:));
     
     iter= 0;
     for iter = 1:max_iter
          %E 步骤
          %求每一样样本对应于GMM函数的输出以及每个高斯分量的输出，
          sigma_old=sigma_i;
          %E步骤。。。。。
          for i=1:Ngauss
          %P(:,i)= Pw(i) * Gauss(data, squeeze(mu(:,i)), squeeze(sigma_i(:,:,i)));%每一个样本对应每一个高斯分量的输出
          qq=Gauss(data, mu(:,i), sigma_i(:,:,i));
          P(:,i) = Pw(i) * qq;
          end
          s=sum(P,2);%
        for j=1:Num
            P(j,:)=P(j,:)/s(j);
        end
       %%%Max步骤
        Pw(1:Ngauss) = 1/Num*sum(P);%权重的估计
        %均值的估计
        for i=1:Ngauss
            sum1=0;
            for j=1:Num
             sum1=sum1+P(j,i).*data(:,j);
            end
          mu(:,i)=sum1./sum(P(:,i));
        end
       
        %方差估计按照公式类似
         %sigma_i
         if((sum(sum(sum(abs(sigma_i- sigma_old))))<min_improve))
             break;
        end
        
        
     end
    
     
end
mu_transpose=mu'; 
row_list=zeros(1,clusters);
for i=1:clusters
%     for p=1:k
    [~,row]=min(sqrt(sum((elit(:,:,2)-repmat(mu_transpose(i,:),size(elit(:,1,2)))).^2,2))); 
    row_list(i)=row;
    elit(row,:,2)=mu_transpose(i,:);
end
