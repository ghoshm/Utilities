%% Florets 
    % Should try coloring each cluster using posterior probability bins
figure; 
nbins = 10; 
hold on; 
scatter(score(:,1),score(:,2),'.k'); 
for k = numComp(1):-1:1  % for each cluster 
    clear scrap scrap_p; 
    scrap = score(idx_numComp_sorted{1,1}==k,1:2); 
    scrap_p = P{1,1}(idx_numComp_sorted{1,1}==k,:); 
    [~,threshold] = histcounts(scrap_p,nbins); 
    
    for p = 1:size(threshold,2)-1 % for each probability bin 
        try 
            scatter(scrap(scrap_p >= threshold(p) & scrap_p < threshold(p+1),1),...
                scrap(scrap_p >= threshold(p) & scrap_p < threshold(p+1),2),...
                '.','markerfacecolor',cmap_cluster{1,1}(k,:)+(1-cmap_cluster{1,1}(k,:))*(1-(1/(size(threshold,2)-p)^.5)),...
                'markeredgecolor',cmap_cluster{1,1}(k,:)+(1-cmap_cluster{1,1}(k,:))*(1-(1/(size(threshold,2)-p)^.5))); 
        catch 
        end 
    end
end 

%% Sorted Data 

% Newer 
scrap = [wake_cells(:,3:end) idx_numComp_sorted{1,1}]; 
scrap = sortrows(scrap);  

% Old 
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

%% Tsne 
% Settings
sample_size = 1000; % Samples to take from each cluster
ex = [2 4 8 16 32 64 128]; % exaggerations to try
per = [3 10 30 90 180 300]; % perplexities to try

sample = []; sample_tags = []; cols = [];
for k = 1:numComp(1) % for each active cluster

    [s,s_t] = datasample(wake_cells_norm(idx_numComp_sorted{1,1}==k,3:end),sample_size,...
        1,'replace',false,'weights',P{1,1}(idx_numComp_sorted{1,1}==k));
    sample = [sample ; s];
    sample_tags = [sample_tags ; s_t];
    cols = [cols ; repmat(cmap_cluster{1,1}(k,:),[sample_size,1])];
    
end

tsne_pro = cell(size(ex,2),size(per,2)); % structure
tsne_time = nan(size(ex,2),size(per,2)); % structure
ex_count = 1; counter = 1; % start counters
for e = ex  % for a range of exaggerations
    per_count = 1;
    for p = per % for a range of perplexities
        tic
        tsne_pro{ex_count,per_count} = tsne(sample,'Algorithm','barneshut',...
            'Exaggeration',e,'NumDimensions',2,'NumPCAComponents',0,...
            'Perplexity',p,'Standardize',1,'Verbose',0);
        tsne_time(ex_count,per_count) = toc;
        disp(horzcat('Finished e = ',num2str(ex_count),...
            ' p = ',num2str(per_count),'. Time Taken = ',...
            num2str(tsne_time(ex_count,per_count)/60),' mins'));
        
        % Figure
        subplot(size(ex,2),size(per,2),counter);
        title(horzcat(num2str(e),' e & ',num2str(p),' p'));
        scatter(tsne_pro{ex_count,per_count}(:,1),...
            tsne_pro{ex_count,per_count}(:,2),[],cols,'filled');
        drawnow; 
        per_count = per_count + 1; % add to counter
        counter = counter + 1; % add to counter 
        
    end
    ex_count = ex_count + 1; % add to counter
end
