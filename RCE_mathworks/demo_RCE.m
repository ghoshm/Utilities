% An example run of the algorithm using Iris Dataset

clear all
close all;

% load iris dataset
load iris_dataset
X = irisInputs;
N = size(irisInputs,2);

% set the fuzzifier constant to 1.4
m = 1.4;

% Optimize the swarm using 80% resampling rate and mahalanobis distance
swarm = RCE(X, 3, 'distance','mahalanobis','fuzzifier',m, 'display','text', ...
'swarm',6, 'subsprob',0.03, 'maxiter',100,'resampling_rate',0.8,'calculate_labels', false);

% calculate the fuzzy labels, crisp labels, and numeric labels from the
% input vectors using the Swarm
[softlabels, crisplabels, numlabels] = swarm_cluster(X,swarm);

% plot the fuzzy voronoi cells on the 1st and 3rd dimension
visualize_swarm(X,swarm,1,3,m,200)

% Perform fuzzy evidence accumulation on the swarm
ensemble = EnsembleAggregate(softlabels,'average',true);

% plot the scatter matrix             
figure('name','scatterplot');
gplotmatrix(X(1:4,:)',[],ensemble.ensemble_labels)
