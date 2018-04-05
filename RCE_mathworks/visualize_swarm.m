function visualize_swarm(in,swarm,d1,d2,m,n_steps)
% visualize the swarm using plot_fuzzy_voronoi and gplotmatrix
% in = inputs
% history = the swarm RCE history output
% d1 = index of the first dimension
% d2 = index of the second dimension
% m = fuzzifier constant, set to 2 if unsure
% n_steps, set to around 50 to 100

% find the subswarm with the minimum average distortion
[~,I] = min(swarm.fitness_graph.average_distortion(:,end));

% plot fuzzy voronoi cells
for i = 1:length(swarm.minima)
    if(swarm.distance == 10)
        Sigma_inv = swarm.Sigma_inv{i}([d1,d2],[d1,d2],:);
    else
        Sigma_inv = false;
    end
    plot_fuzzy_voronoi(in([d1,d2],:),swarm.minima{i}([d1,d2],:),m, ...
                 n_steps,swarm.distance,1,Sigma_inv);
end
axis off;


blue = [linspace(1,0,100)',linspace(1,0,100)',ones(100,1)];

softlabels = swarm_cluster(in,swarm);
% plot the fuzzy membership matrix
y_tick = cumsum(double(swarm.numclust));
y_tick = 0.5+y_tick - y_tick(1);
ticklabels = arrayfun(@(i) sprintf('Subswarm %d',i),1:length(softlabels),'uniformoutput',false);

figure('name','fuzzy membership');
imagesc(cell2mat(softlabels')); colormap(blue);
set(gca,'ytick',y_tick,'ygrid','on','gridlinestyle','-','yticklabel',ticklabels);

