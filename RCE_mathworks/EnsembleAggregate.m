function stats = EnsembleAggregate(softlabels,method,visualize)
% Perform fuzzy evidence accumulation to get the consensus of the swarm.
% Note that for larger data, CA-tree compression needs to be used. The code
% for CA-tree will be posted in the near future.
%
% softlabels = fuzzy labels of inputs
% method = 'average', 'single', 'ward', or 'complete'
% visualize = true/false, decide whether to print the figures
%
% 
% outputs:
% ensemble_labels = the ensemble partition suggested by maximum lifetime
% tree = the hierarchical cluster tree, visualize using dendrogram
% threshold = (suggested) maximum lifetime threshold
% lifetimes = the cluster lifetimes
% consensus_matrix = the consensus matrix
% item_consensus = the item consensus (the average consensus of an item
%                  w.r.t. elements in another cluster after ensemble
%                  partition
% cluster_consensus = the average consensus of between clusters


% Get the normalized affinity matrix
U = cell2mat(softlabels');
A = U'*U;
D = diag(diag(A));
A = D^-0.5 * A * D^-0.5;

% construct the dendrogram
DistanceMatrix = 1-A;
DistanceMatrix(eye(size(A))==1) = 0;
Z = linkage(squareform(DistanceMatrix),method);
N = size(DistanceMatrix,1);

% compute the maximum lifetime cut
lifetimes = diff(Z(:,3));
[~, Ith] = max(lifetimes);

th = max(Z(Ith,3) + lifetimes(Ith)*0.5,eps);

% apply the cut to the dendrogram
ensemble_labels = cluster(Z,'cutoff',th,'criterion','distance');

% calculate the item and cluster consensus
[i,j] = meshgrid(1:max(ensemble_labels));
An = A - eye(size(A));
item_consensus = arrayfun(@(i,j) (sum(An(ensemble_labels == j,ensemble_labels == i),1))/nnz(ensemble_labels==j), i,j,'uniformoutput',false);

cluster_consensus = cellfun(@mean, item_consensus, 'uniformoutput',false);

stats.ensemble_labels = ensemble_labels;
stats.tree = Z;
stats.threshold = th;
stats.lifetimes = lifetimes;
stats.consensus_matrix = A;
stats.item_consensus = item_consensus;
stats.cluster_consensus = cluster_consensus;


if(visualize)

    blue = [linspace(1,0,100)',linspace(1,0,100)',ones(100,1)];
    pasteljet = bsxfun(@plus,[0.1, 0.1, 0.1], jet*0.7);
    
    figure('name','Cluster Dendrogram');
    subplot('position',[0.0500    0.8178    0.9000    0.1322]);
    [~,~,perm] = dendrogram(Z,N);
    dendrogram(Z,N,'colorthreshold',th);

    line(get(gca,'xlim'), [th th]);
    text(1,th,'maximum lifetime cut','verticalalignment','bottom');

    axis off;
    subplot('position',[0.0500    0.7894    0.9000    0.0184]);
    image(label2rgb(ensemble_labels(perm)')); 
    axis off;

    subplot('position', [0.0500    0.0500    0.9000    0.7294]);
    imagesc(A(perm,perm) ); 
    colormap(blue)
    axis off
    
    figure('name','item consensus and cluster consensus');
    subplot(211);
    bar(cell2mat(item_consensus)','stacked'); 
    title('Item Consensus');
    xlim([0.5,N]);
    colormap(pasteljet);

    %
    % compute the cluster consensus
    subplot(212);
    bar(cell2mat(cluster_consensus)); 
    title('Cluster Consensus');
end
