%% Florets 

cmap = lbmap(4,'redblue'); 
threshold = (0:10:100)/100;
hold on; 
scatter(X{1}(:,1),X{1}(:,2),'.k'); 
for k = 1:4
    clear scrap scrap_p; 
    scrap = X{1,1}(idx{1}==k,:); 
    scrap_p = P{1,1}(idx{1}==k,:); 
    
    for p = 1:10
        try 
            scatter(scrap(scrap_p >= threshold(p) & scrap_p < threshold(p+1),1),...
                scrap(scrap_p >= threshold(p) & scrap_p < threshold(p+1),2),...
                '.','markerfacecolor',cmap(k,:)+(1-cmap(k,:))*(1-(1/(9-p)^.5)),...
                'markeredgecolor',cmap(k,:)+(1-cmap(k,:))*(1-(1/(11-p)^.5))); 
        catch 
        end 
    end
end 

%% Sorted Data 
wake_cells_norm = [wake_cells_norm  idx{1}];
[~,O] = sort(wake_cells_norm(:,end));
wake_cells_norm = wake_cells_norm(O,:);

for k = 1:4 
    clear O scrap; 
    scrap = wake_cells_norm(wake_cells_norm(:,end) == k,:); 
    [~,O] = sort(scrap(:,1)); 
    wake_cells_norm(wake_cells_norm(:,end) == k,:) = scrap(O,:);
end 

figure; 
imagesc(wake_cells_norm,[-1 4]) 