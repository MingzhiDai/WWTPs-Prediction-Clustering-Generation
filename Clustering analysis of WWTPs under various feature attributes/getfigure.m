
function a=getfigure(pop,label,data)

% data=data(:,1:2);
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
    scatter(data(a,2),data(a,6),'filled') 
    if k==1
        plot(data(a,2),data(a,6),'o','color',color(k,:),'MarkerSize',8);
    elseif k==2,
        plot(data(a,2),data(a,6),'o','color',color(k,:),'MarkerSize',8);
    elseif k==3
        plot(data(a,2),data(a,6),'o','color',color(k,:),'MarkerSize',8);
    elseif k==4
        plot(data(a,2),data(a,6),'o','color',color(k,:), 'MarkerSize',8);
    elseif k==5
        plot(data(a,2),data(a,6),'o','color',color(k,:), 'MarkerSize',8);
    elseif k==6
        plot(data(a,2),data(a,6),'o','color',color(k,:),'MarkerSize',8);
    elseif k==7
        plot(data(a,2),data(a,6),'o','color',color(k,:),'MarkerSize',8);
    elseif k==8
        plot(data(a,2),data(a,6),'o','color',color(k,:),'MarkerSize',8);
    elseif k==9
        plot(data(a,2),data(a,6),'o','color',color(k,:),'MarkerSize',8);
    elseif k==10
        plot(data(a,2),data(a,6),'o ','color',color(k,:), 'MarkerSize',8);
    elseif k==11
        plot(data(a,2),data(a,6),'o', 'color',color(k,:),'MarkerSize',8);
    elseif k==12
        plot(data(a,2),data(a,6),'o','color',color(k,:),'MarkerSize',8);
    elseif k==13
        plot(data(a,2),data(a,6),'o','color',color(k,:),'MarkerSize',8);
    elseif k==14
        plot(data(a,2),data(a,6),'o','color',color(k,:),'MarkerSize',8);
    elseif k==15
        plot(data(a,2),data(a,6),'o ','color',color(k,:), 'MarkerSize',8);
    elseif k==16
        plot(data(a,2),data(a,6),'o', 'color',color(k,:),'MarkerSize',8);
    elseif k==17
        plot(data(a,2),data(a,6),'o','color',color(k,:),'MarkerSize',8);
    elseif k==18
        plot(data(a,2),data(a,6),'o','color',color(k,:),'MarkerSize',8);
    elseif k==19
        plot(data(a,2),data(a,6),'o','color',color(k,:),'MarkerSize',8);
    else
        plot(data(a,2),data(a,6),'o', 'color',color(k,:),'MarkerSize',8);
    end
    hold on;
end

hold off;

return;