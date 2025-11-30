% 高斯分布数据集
mu = [2 3];
SIGMA = [1 0; 0 2];
r = mvnrnd(mu,SIGMA,100);
plot(r(:,1),r(:,2),'r+');
hold on;
mu = [7 8];
SIGMA = [ 1 0; 0 2];
r2 = mvnrnd(mu,SIGMA,100);
plot(r2(:,1),r2(:,2),'*');
hold off;