function a=getfigure_3(pop,label,data)
% data=data(:,[2,4]);
color=[
     [255 0 0]/255;
    [0 255 0]/255;
    [0 0 255]/255;
    [128 128 0]/255;
    [128 0 128]/255;
    [0 128 128]/255;
    [128 128 64]/255;
    [64 128 128]/255;
    [128 64 128]/255;
    [64 64 64]/255;
    [255 255 64]/255;
    [255 64 255]/255;
    [64 255 255]/255;
    [64 64 64]/255;
    [64 0 0]/255;
    [0 64 0]/255;
    [0 0 64]/255;
    [64 64 128]/255;
    [64 128 64]/255;
    [128 64 64]/255;
    ];


for k=1:20
    a=find(label==k);
    if k==1
        scatter3(data(a,2),data(a,6),data(a,8),'filled');                    
    elseif k==2
        scatter3(data(a,2),data(a,6),data(a,8),'filled');         
    elseif k==3
        scatter3(data(a,2),data(a,6),data(a,8),'filled');
    elseif k==4
        scatter3(data(a,2),data(a,6),data(a,8),'filled');
    elseif k==5
        scatter3(data(a,2),data(a,6),data(a,8),'filled');
    elseif k==6
        scatter3(data(a,2),data(a,6),data(a,8),'filled');
    elseif k==7
        scatter3(data(a,2),data(a,6),data(a,8),'filled');
    elseif k==8
        scatter3(data(a,2),data(a,6),data(a,8),'filled');
    elseif k==9
        scatter3(data(a,2),data(a,6),data(a,8),'filled');    
    elseif k==10
        scatter3(data(a,2),data(a,6),data(a,8),'filled');
    elseif k==11
        scatter3(data(a,1),data(a,2),data(a,3),'filled');
    elseif k==12
        scatter3(data(a,1),data(a,2),data(a,3),'filled');           
    elseif k==13
        scatter3(data(a,1),data(a,2),data(a,3),'filled');
    elseif k==14
        scatter3(data(a,1),data(a,2),data(a,3),'filled');
    elseif k==15
        scatter3(data(a,1),data(a,2),data(a,3),'filled');   
    elseif k==16
        scatter3(data(a,1),data(a,2),data(a,3),'filled');
    elseif k==17
        scatter3(data(a,1),data(a,2),data(a,3),'filled');
    elseif k==18
        scatter3(data(a,1),data(a,2),data(a,3),'filled');
    elseif k==19
        scatter3(data(a,1),data(a,2),data(a,3),'filled');
    else
        scatter3(data(a,1),data(a,2),data(a,3),'filled');
    end
    hold on;
end

hold off;

return;
