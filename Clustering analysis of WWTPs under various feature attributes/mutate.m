function [pop] = mutate(pop,pm)       
for i = round(NP/2) : NP                
    %MutationRate = pmu(i) * (1 - Prob(i) / Pmax);            
    MutationRate = pmu(i) ;            
    for p=1:k                
        for q = 1 : c                    
            if MutationRate> rand                        
                pop(p,q,i) = rand*abs(d_up(q)-d_down(q))+d_down(q);                    
            end            
        end        
    end    
end