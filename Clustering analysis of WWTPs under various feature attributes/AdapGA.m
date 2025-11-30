function [pop]=AdapGA(pop,cost,data,k,NP,MAXGEN,Pc1,Pc2,Pm1,Pm2)
%自适应遗传算法
% L = ceil(log2((b-a)/eps+1));
[~,c]=size(data);
A=9.903438;
SelFather=0;
GA_pop=zeros(k,c,NP);
x(k,c,NP)= pop(k,c,NP);
for i=1:NP    
    fx(i) = cost(i);    
end
for k=1:MAXGEN
    sumfx = sum(fx);   
    Px = fx/sumfx;
    PPx = 0;   
    PPx(1) = Px(1);    
    for i=2:NP        
        PPx(i) = PPx(i-1) + Px(i);        
    end
    for i=1:NP      
        sita = rand();       
        for n=1:NP           
            if sita <= PPx(n)                
                SelFather = n;
            else
                SelFather = n-1;               
            end           
        end
    end
      
    for i=1:NP        
        favg = sumfx/NP;       
        fmax = max(fx);
        Selmother = round(rand()*(NP-1))+1;      
        posCut = round(rand()*(c-2)) + 1;             
        Fitness_f = fx(SelFather);        
        Fitness_m = fx(Selmother);       
        Fm = min(Fitness_f,Fitness_m);        
        if Fm<=favg            
%             Pc = Pc1*(fmax - Fm)/(fmax - favg); 
              Pc = (Pc2-Pc1)*cos((fmax - Fm)*pi/(fmax - favg))/(1+exp(A*(2*(fmax - Fm)/(fmax - favg))))+Pc1;
%               Pc = (Pc2-Pc1)/(1+exp(A*(2*(fmax - Fm)/(fmax - favg))))+Pc1;
        else            
            Pc = Pc2;            
        end       
        r1 = rand();       
        if r1<=Pc
            GA_pop(:,1:posCut,i) = x(:,1:posCut,SelFather);          
            GA_pop(:,(posCut+1):c,i) = x(:,(posCut+1):c,Selmother);            
            fmu = cost(i);
            if fmu<=favg
%                 Pm = Pm1*(fmax - fmu)/(fmax - favg);
                  Pm = (Pm2-Pm1)*cos((fmax - Fm)*pi/(fmax - favg))/(1+exp(A*(2*(fmax - Fm)/(fmax - favg))))+Pm1;
%                   Pm = (Pm2-Pm1)/(1+exp(A*(2*(fmax - Fm)/(fmax - favg))))+Pm1;
            else
                Pm = Pm2;
            end           
            r2 = rand();           
            if r2 <= Pm               
                posMut = round(rand()*(c-1) + 1);            
                GA_pop(:,posMut,i) = ~GA_pop(:,posMut,i);                                  
            else          
                GA_pop(:,:,i) = x(:,:,SelFather);
            end
        end
    end
end
pop = GA_pop;    

    


