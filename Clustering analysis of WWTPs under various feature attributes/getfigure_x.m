function a=getfigure(pop,label,data)

data=data(:,1:2);
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
    
    ];


for k=1:10
    a=find(label==k);
    if k==1
        plot(data(a,1),data(a,2),'x', 'color',color(k,:),'MarkerSize',8);
    elseif k==2
        plot(data(a,1),data(a,2),'o', 'color',color(k,:),'MarkerSize',8);
        elseif k==3
        plot(data(a,1),data(a,2),'v', 'color',color(k,:),'MarkerSize',8);
        elseif k==4
        plot(data(a,1),data(a,2),'p','color',color(k,:), 'MarkerSize',8);
        elseif k==5
        plot(data(a,1),data(a,2),'+ ','color',color(k,:), 'MarkerSize',8);
        elseif k==6
        plot(data(a,1),data(a,2),'x', 'color',color(k,:),'MarkerSize',8);
        elseif k==7
        plot(data(a,1),data(a,2),'o','color',color(k,:),'MarkerSize',8);
        elseif k==8
        plot(data(a,1),data(a,2),'v','color',color(k,:),'MarkerSize',8);
        elseif k==9
        plot(data(a,1),data(a,2),'p','color',color(k,:),'MarkerSize',8);
    else
        plot(data(a,1),data(a,2),'+', 'color',color(k,:),'MarkerSize',8);
    end
    hold on;
end

hold off;

return;