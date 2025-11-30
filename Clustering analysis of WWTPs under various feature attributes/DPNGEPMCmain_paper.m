% EPMC clustering 基于情感偏好的迁徙聚类算法
function DPNGEPMCmain_paper(Input_File, Microbial_hierarchy, Feature_attribute, Microbial_feature_number, Output_folder)
%     disp("Program running...");
%     disp(Input_File);
% %     disp(Output_folder);
%     Input_File = 'D:\BaiduNetdiskDownload\clustering\WWTP--environmental and geographical features\Phylum\WWTP_Phylum--environmental and geographical features.csv';
%     Microbial_hierarchy = 'Phylum';
%     Feature_attribute = 'environmental and geographical';
%     Microbial_feature_number = 21;
%     Output_folder = 'D:\BaiduNetdiskDownload\clustering\WWTP--environmental and geographical features\Phylum';
    tic;
    NP=15; %个体数量
    NP_split=1;
    clusters=3;  %原数据集标签数
    k=5;  % 聚类数目
    run=1;
    iteration=100;% 迭代次数
    sigma=5000;
    d=5;
    eta=0.8;
    c3=1;
    x=0.8;
    alpha=0.8;
    beta=0.2;
    w_start=0.9;
    w_end=0.4;
    I = 1; % max immigration rate for each island
    E = 1; % max emigration rate, for each island
    % dt=1;
    kchange=ones(1,iteration);  %聚类簇数矩阵（1x100)
    global label;% 类标签
    global pmu;% 变异概率
    % global pmodify;% 学习概率。
    global cost;% 个体代价
    global d_up;
    global d_down;
    % global Prob;
    afa=0.7;
    elitnum=2;
    elitcost=zeros(1,elitnum);
    global SpeciesCount;
    % numberhhh=zeros(1,elitnum);
    W=zeros(1,100);
    T=zeros(1,100);
    n_feature = 21;
    
    % for count=1:10
     
    % filename='C:\Users\Dai MZ\Downloads\148个整理好的UCI数据集及相应代码\148个整理好的UCI数据集及相应代码\iris\预处理完成\iris.mat';
    % filename='D:\U28XW\自适应指令集资源调度匹配优化_算法模拟\自适应指令集资源调度算法匹配_需求属性模拟数据.csv';
    % filename='F:\optimization\Matlab_code\EPMC_probilityAccess1_Simulation_GLEPMC\UCI dataset\Soybean.mat';
    % filename='F:\optimization\Matlab_code\EPMC_probilityAccess1_Simulation_GLEPMC\UCI dataset\wine.mat';
    % filename='C:\Users\Dai MZ\Downloads\148个整理好的UCI数据集及相应代码\148个整理好的UCI数据集及相应代码\seeds\预处理完成\seeds.mat';
    % filename='C:\Users\Dai MZ\Downloads\148个整理好的UCI数据集及相应代码\148个整理好的UCI数据集及相应代码\glass\预处理完成\glass.mat';
    % filename='C:\Users\Dai MZ\Downloads\148个整理好的UCI数据集及相应代码\148个整理好的UCI数据集及相应代码\haberman-survival\预处理完成\haberman-survival.mat';
    % filename='C:\Users\Dai MZ\Downloads\148个整理好的UCI数据集及相应代码\User Knowledge.mat';
    % filename='C:\Users\Dai MZ\Downloads\148个整理好的UCI数据集及相应代码\148个整理好的UCI数据集及相应代码\vowel\预处理完成\vowel.mat';
    % filename='C:\Users\Dai MZ\Downloads\148个整理好的UCI数据集及相应代码\148个整理好的UCI数据集及相应代码\blood\预处理完成\blood.mat';
    % filename='C:\Users\Dai MZ\Downloads\148个整理好的UCI数据集及相应代码\148个整理好的UCI数据集及相应代码\banknote\预处理完成\banknote.mat';
    % filename='C:\Users\Dai MZ\Downloads\148个整理好的UCI数据集及相应代码\148个整理好的UCI数据集及相应代码\car\预处理完成\car.mat';
    filename='D:\BaiduNetdiskDownload\clustering\WWTP--environmental and geographical features\Phylum\WWTP_Phylum--environmental and geographical features.csv';
    % filename='E:\all_data\csv_500\500_0_1.mat';
    % filename='E:\all_data\csv_2048\2048_0_9.mat';
    
    % filename='D:\paper\148个整理好的UCI数据集及相应代码\148个整理好的UCI数据集及相应代码\adult\预处理完成\adult_train.mat';
    % filename='D:\paper\148个整理好的UCI数据集及相应代码\148个整理好的UCI数据集及相应代码\bank\预处理完成\bank.mat';
    % Data=load(strcat(filename,'iris.mat'));
    
    % Data=load(filename);
    % all_data=Data.Soybean; % Soybean DataSet
    
    all_data=readtable(Input_File);
    header = all_data.Properties.VariableNames;
    
    [ro,length]=size(all_data);
    % all_data=Data.minmax_scaling; 
    % num = xlsread('E:\360MoveData\Users\Dai MZ\Documents\WeChat Files\wxid_brkhvc2wfkbp21\FileStorage\File\2021-04\114309734_1_儿童伴读机器人用户信息调研_13_13.xlsx');
    % num = xlsread('E:\360MoveData\Users\Dai MZ\Documents\WeChat Files\wxid_brkhvc2wfkbp21\FileStorage\File\2021-04\数据2.xlsx');
    % data=num;
    % all_data=Data.minmax_scaling;
    % datalabel=load(strcat(filename,'data_label.mat'));
    % data=all_data(:,2:length);  %length
    
    data=table2array(all_data(:,2:length));
    % [~,Sorted_data] = sort(data, 1);
    
    % A=B';
    % data1=mapminmax(A,0,1);
    % data=data1';
    % datalabel=all_data(:,1);
    [~,length_header] = size(header);
    datalabel = header(2:length_header)';
    datalabel = string(datalabel);
    
    fitness=zeros(1,run);
    convercost=zeros(run,iteration);
    time=zeros(1,run);
    accuracy=zeros(1,run);
    classnum=zeros(1,run);
    finalcost=zeros(1,run);
    finalCost2=zeros(1,run);
    
    for runcount=1:run
    
        [r,c]=size(data);
        Pop=zeros(k,c,NP);
        H=zeros(k,c,NP);
        J=zeros(k,c,NP);
    
        feel=ones(NP,NP);  %feel = unify(feel);
        scale=zeros(NP,NP);
        [pop,Antenna_L,Antenna_R,pmu,pmodify,d_up,d_down,elit,individual]=initial(NP,k,data,elitnum);
        label=zeros(NP,c);
        for j=1:NP    %第一次贴标签 %随机的第一个个体第一行（某个簇中心点）依次与data每一行比较。
                      %得出点和那个中心近就属于哪个聚类，并打标签
            for i=1:r
                label(j,i)=1;
                for m=2:k
                    if norm(pop(m,:,j)-data(i,:))<norm(pop(label(j,i),:,j)-data(i,:))
                        % if(sqrt(sum((pop(m,:,j)-iris(i,:)).^2))<sqrt(sum((pop(label(i),:,j)-iris(i,:)).^2)))
                        label(j,i)=m;
                    end
                end
            end
    
            cost(j)=caculateCost(pop(:,:,j),label(j,:),data,k);
        end  % end NP
        [pop,cost,individual,feel]=popSort(pop,NP,cost,label,individual,feel); %15个聚类个体，个体数，15个fitness,
                                                              %15种聚类所有data类别标签，15个个体，15个个体相互的feel
    
        % figure((runcount-1)*10+1);
        [cost(1),label(1,:)]=caculateCost(pop(:,:,1),label(1,:),data,k); 
    
    
       %initialk=getK(pop(:,:,1),label(1,:),data,k); %提取出初始聚类数目
        initialk=k;
        initialk
    %     getfigure(pop(:,:,1),label(1,:),data);
    %     saveas(gcf, 'output1', 'fig')
    
        for t=1:iteration
    %         filename
            disp(['***************','运行次数:',num2str(runcount),'/',num2str(run) ,' ***************','迭代次数: ',num2str(t),' ***************'])
            NP_split=floor(NP*t/iteration);  %向下取整
           %NP_split=floor(NP/2);
           %matlabpool local 8
            for i=1:elitnum %精英个体
                elit(:,:,i)=pop(:,:,i);
                elitcost(i)=cost(i);
            end  
            
    %         a=exp(-t);
    %         [elit]=LVQ(elit,a,r,elitnum,data,datalabel,label,k,clusters);
    %         [elit]=EM_GMM(elit,data,clusters);
            
           
            for i = 1 : NP  % 计算每个栖息地的被选为目标个体的概率  %3.1 Migration model_1
                SpeciesCount(i) = NP +1- i;
                lambda(i) = I * (1 -SpeciesCount(i) / NP);% Compute immigration rate and extinction rate for each individual.
                mu(i) = E * SpeciesCount(i) /NP;
                %Prob(j) = 1 / NP;
            end
           
            SelectIndex = zeros(1,NP);
            for i = 1 : NP  %3.2 Emotional preference model
    
                if rand > pmodify(i)
                    continue;
                end
                if sum(feel(i,:))~=0
                    scale(i,:)=afa*feel(i,:)/sum(feel(i,:))+(1-afa)*mu/sum(mu);
                else
                    temp1=feel(i,:)-min(feel(i,:))*ones(1,NP);
                    scale(i,:)=afa*-temp1/sum(temp1)+(1-afa)*mu/sum(mu);
                end
    
                RandomNum =rand * sum(scale(i,:));
                Select = scale(i,1);
                SelectPositon= 1;%下标位置
                while (RandomNum > Select) && (SelectPositon < NP)
                    SelectPositon =  SelectPositon+1;
                    Select = Select + scale(i, SelectPositon);
                end
               %SelectIndex(i) = individual(SelectPositon);
                SelectIndex(i) = SelectPositon;
                temp_i=pop(:,:,i);
                temp_labeli=label(i,:);
                temp_costi=cost(i);
                % Normalize the immigration rate.
                % lambdaScale =  (lambda(i) - lambdaMin) / (lambdaMax - lambdaMin);
                % Probabilistically input new information into habitat i
                for p=1:k             %3.3 Information learning
                    for j = 1 : c
                       %if rand < lambda
                        if rand < lambda(i)
                          % Pick a habitat from which to obtain a feature                          
                            RandomNum =rand * sum(mu);
                            Select = mu(1);
                            SelectPositon= 1;%下标位置
                            while (RandomNum > Select) && (SelectPositon < NP)
                                SelectPositon =  SelectPositon+1;
                                Select = Select + mu(SelectPositon);
                            end
    
                            if (i<NP_split&&SelectPositon>NP_split)||(i>NP_split&&SelectPositon<NP_split)
                                % pop(p,j,i) = min(pop(p,j,SelectIndex(i)),pop(p,j,i))+abs(pop(p,j,SelectIndex(i))-pop(p,j,i))*rand;
                                % pop(p,j,i) =pop(p,j,i)+rand*(pop(p,j,i) - pop(p,j,SelectIndex(i)))+rand*(pop(p,j,i) - pop(p,j,1));
                                % pop(p,j,i) =pop(p,j,i)+rand*(pop(p,j,i) - pop(p,j,1))+rand*(pop(p,j,i) - pop(p,j,SelectIndex(i)));
                                 pop(p,j,i) =pop(p,j,i)+rand*(pop(p,j,i) - pop(p,j,1));                           
                                 
                            else
                                 % pop(p,j,i) =pop(p,j,i)+0.8*(pop(p,j,i) - pop(p,j,SelectIndex(i)))+0.2*(pop(p,j,i) - pop(p,j,SelectPositon));
                                
                                 sigma = eta * sigma;
                                 d = sigma/c3;
                                 Antenna_R = Antenna_R + (x*(alpha * (pop(p,j,i) - pop(p,j,SelectIndex(i)))+ beta*(pop(p,j,i) - pop(p,j,SelectPositon))))*d/2;
                                 Antenna_L = Antenna_L - (x*(alpha * (pop(p,j,i) - pop(p,j,SelectIndex(i)))+beta*(pop(p,j,i) - pop(p,j,SelectPositon))))*d/2;
                                 H(p,j,i) = sigma* (alpha*(pop(p,j,i) -pop(p,j,SelectIndex(i)))+beta*(pop(p,j,i) - pop(p,j,SelectPositon)))* Sign(Antenna_L,Antenna_R);
                                 T(t)=t;
                                 w = 0.4 - 0.5 * sin(pi / 2 * sqrt((1 - t/iteration)^3));                           
                                 W(t)=w;
                                 Pop(p,j,i) = pop(p,j,i);
                                 pop(p,j,i) = pop(p,j,i) + w*J(p,j,i) +x*(alpha*(pop(p,j,i) - pop(p,j,SelectIndex(i)))+beta*(pop(p,j,i) - pop(p,j,SelectPositon)))+(1-x)*H(p,j,i);
                                 J(p,j,i) = alpha*( Pop(p,j,i) - pop(p,j,SelectIndex(i))) + beta*( Pop(p,j,i) - pop(p,j,SelectPositon));
                            end
    
                                % pop(p,j,i) = pop(p,j,SelectIndex);
                                % pmodify(i)=min(pmodify(i),pmodify(SelectIndex(i)))+abs(pmodify(i)-pmodify(SelectIndex(i)))*rand;
                                % pmu(i)=min(pmu(i),pmu(SelectIndex(i)))+abs(pmu(i)-pmu(SelectIndex(i)))*rand;
                        end
    
                    end
                end
    
                %计算前后好坏
                for ybsize=1:r
                    label(i,ybsize)=1;
                    for m=2:k
                        if norm(pop(m,:,i)-data(ybsize,:))<norm(pop(label(i,ybsize),:,i)-data(ybsize,:))
                            %if(sqrt(sum((pop(m,:,j)-iris(i)).^2))<sqrt(sum((pop(label(i),:,j)-iris(i)).^2)))
                            label(i,ybsize)=m;
                        end
                    end
                end
                cost(i)=caculateCost(pop(:,:,i),label(i,:),data,k);
    
                if cost(i)>temp_costi%cost变大，效果变差，好感度减少1次
    
                    if rand < 1-t/iteration
                        pop(:,:,i)=temp_i;
                        label(i,:)=temp_labeli;
                        cost(i)=temp_costi;
                    end
                    feel(i,SelectIndex(i))=feel(i,SelectIndex(i))-1;
                else
                    feel(i,SelectIndex(i))=feel(i,SelectIndex(i))+1;
                end
            end % end modify
    
            [pop,cost,individual,feel]=popSort(pop,NP,cost,label,individual,feel);
    
            % Mutation        %3.4 Inertial learningmodel
            % Mutate only the worst half of the solutions
            %Pmax = max(Prob);
            AdapGA(pop,cost,data,k,NP,500,0.5,0.8,0.005,0.05);
            for i = round(NP/2) : NP     
    %             MutationRate = pmu(i) * (1 - Prob(i) / Pmax);
                MutationRate = pmu(i);
                for p=1:k
                    for q = 1 : c
                        if MutationRate> rand
                            pop(p,q,i) = rand*abs(d_up(q)-d_down(q))+d_down(q);
                        end
                    end
                end
            end
    %         
            %%%%%%%一维线性栅格距离帕雷托最优个体补充
            
    %          for i = round(NP*13/15) : NPe     
    %               for p=1:k
    %                     for q = 1 : c
    %                          pop(p,q,i) = pop(p,q,NP-i+1);
    %                     end
    %               end
    %          end
    %          [pop,cost,individual,feel]=popSort(pop,NP,cost,label,individual,feel);
             
            % 变异之后，给个体按照代价从低到高排序，更换代价最高的elitnum个个体
            for j=1:round(NP/2) : NP
                for i=1:r
                    label(j,i)=1;
                    for m=2:k
                        if norm(pop(m,:,j)-data(i,:))<norm(pop(label(j,i),:,j)-data(i,:))
                       %if(sqrt(sum((pop(m,:,j)-iris(i)).^2))<sqrt(sum((pop(label(i),:,j)-iris(i)).^2)))
                            label(j,i)=m;
                        end
                    end
                end
                cost(j)=caculateCost(pop(:,:,j),label(j,:),data,k);
            end  % end NP
            [pop,cost,individual,feel]=popSort(pop,NP,cost,label,individual,feel);
           
             
            for i=1:elitnum
                pop(:,:,NP-elitnum+i)=elit(:,:,i);
                cost(NP-elitnum+i)=elitcost(i);
            end
            
    
            [pop,cost,individual,feel]=popSort(pop,NP,cost,label,individual,feel);
            % min(cost)%最小值
            convercost(runcount,t)=min(cost); % converge
    
            kchange(t)=getK(pop(:,:,1),label(1,:),data,k); 
            [cost(1),label(1,:)]=caculateCost(pop(:,:,1),label(1,:),data,k); 
            
    %        
    % %      if t==1
    % %         figure((runcount-1)*10+1);
    % % 
    % %         getfigure(pop(:,:,1),label(1,:),data);
    % % 
    % %         saveas(gcf, 'output2', 'fig')
    % %     elseif t==5
    % %         figure((runcount-1)*10+6);
    % % 
    % %         getfigure(pop(:,:,1),label(1,:),data);
    % % 
    % %         saveas(gcf, 'output3', 'fig')
    % %      elseif t==10
    % %         figure((runcount-1)*10+11);
    % %         getfigure(pop(:,:,1),label(1,:),data);
    % % 
    % %         saveas(gcf, 'output4', 'fig')
    % %      elseif t==50
    % %         figure((runcount-1)*10+50);
    % %         getfigure(pop(:,:,1),label(1,:),data);
    % % 
    % %         saveas(gcf, 'output5', 'fig')
    % %      elseif t==100
    % %         figure((runcount-1)*10+100);
    % %         getfigure(pop(:,:,1),label(1,:),data);
    % % 
    % %         saveas(gcf, 'output6', 'fig')
    
    
    
            figure((runcount-1)*10+101);
            plot(kchange(1,1:100),'DisplayName','kchange(1,1:100)','YDataSource','kchange(1,1:100)');
    %         saveas(gcf, 10+runcount, 'fig')
        % end
        end
        % tic;
        % runcount
    
    
         % end iteration
     
    
        toc;
        time(runcount)=toc;%--------------------------------------------------------------------------------------------------time
    
        %输出最后结果
        finalcost(runcount)=cost(1);
        %fitness(1,runcount)=cost(1);
        %finalcost
        finalcost(runcount)
        finalcost
        pop(:,:,1)
        %label(1,:)
    
        % accuracy(runcount)=Jaccard(datalabel,label(1,:));    %accuracy
        % accuracy(runcount)
    
        %  finalcost2   提取最终类的中心点，返回cost
        [finalCost2,C,label2]=getFinalCost(data,label(1,:),k);
        % finalCost2
        C %最终的聚类中心点
        [classnum_count,ro]=size(C);
        Microbiota_C = C(:,ro-Microbial_feature_number+1:ro)';
        write_C = [datalabel(ro-Microbial_feature_number+1:ro), Microbiota_C];
        % C_sorted = sortrows(write_C, "Var2", "descend");
        double_C = str2double(write_C(:,2));        % 把第二列转为数字
        [~, idx] = sort(double_C, 'descend'); % 排序得到索引
        C_sorted = write_C(idx, :);             % 按索引重排整个 cell 数组
    
        [nRows, nCols] = size(C_sorted);
        
        % 第一行：第 1 列空，第 2 列开始依次 1,2,3...
        new_header = cell(1, nCols);
        new_header{1} = '';                        % 第一列空
        new_header(2:end) = num2cell(1:(nCols-1)); % 第二列开始写1,2,3...
        
        % 竖向拼接
        C_sorted_with_header = [new_header; C_sorted];
    
    
    
        Data = {};
        Data = sorted_ranking(classnum_count,C,data,label(1,:),k);
        for i= 1:classnum_count
    %         
            xlswrite(fullfile(Output_folder, ['WWTP_' Microbial_hierarchy '--' Feature_attribute '-The WWTP ranking information of each centroid.xlsx']), Data{i},i);
    
    
        end
    %       
            xlswrite(fullfile(Output_folder, ['WWTP_' Microbial_hierarchy '--' Feature_attribute '-Information of each centroid.xlsx']), C_sorted_with_header)
            draw_clustering_heatmap( ...
            fullfile(Output_folder, ['WWTP_' Microbial_hierarchy '--' Feature_attribute '-Information of each centroid.xlsx']), Microbial_hierarchy,...
            fullfile(Output_folder, ['WWTP_' Microbial_hierarchy '--' Feature_attribute '-Microbio heatmap.jpg']) ...
    );
    
    
        
        classnum(runcount)=classnum_count;    %classnum
    end                 






