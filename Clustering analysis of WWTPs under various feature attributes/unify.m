function feel=unify(feel)
[r,c]=size(feel);
for i=1:r
    minValue=min(feel(i,:));
    if minValue<0
        feel(i,:)=feel(i,:)-minValue*ones(1,c);
    end
    feel(i,:)=feel(i,:)/sum(feel(i,:));
end
return;