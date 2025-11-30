function error = fun_error(x,S1,S2,S3,net_optimized,Pn_train,Tn_train)
%该函数用来计算适应度值
%x input 个体
%S1 input 输入层节点数
%S2 input 隐含层节点数
%S3 input 输出层节点数
%net_optimized input 网络
%inputn input 训练输入数据
%outputn input 训练输出数据

%error output 个体适应度值

%提取
W1=x(1:S1*S2);
W2=x(S1*S2+1:S1*S2+S2*S3);
B1=x(S1*S2+S2*S3+1:S1*S2+S2*S3+S2);
B2=x(S1*S2+S2*S3+S2+1:S1*S2+S2*S3+S2+S3);

% 设置训练参数
net_optimized.trainParam.epochs = 3000;
net_optimized.trainParam.show = 100;
net_optimized.trainParam.goal = 0.001;
net_optimized.trainParam.lr = 0.1;

%网络权值赋值
net_optimized.IW{1,1} = reshape(W1,S2,S1);
net_optimized.LW{2,1} = reshape(W2,S3,S2);
net_optimized.b{1} = reshape(B1,S2,1);
net_optimized.b{2} =reshape(B2,S3,1);

% 利用新的权值和阈值进行训练
net_optimized = train(net_optimized,Pn_train,Tn_train);
 
an=sim(net_optimized,Pn_train);
error1=an-Tn_train;
error=mse(error1);
end 