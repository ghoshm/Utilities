function [softlabels, crisplabels,labels] = swarm_cluster(in,swarm)
% [softlabels, labels] = swarm_cluster(in,swarm)
% clusters the input vectors in "in" using the RCE swarm "swarm"
% in = dim x N input vectors
% swarm = the RCE swarm obtained from executing RCE
%
% example usage:
%
% 
% clear all
% close all;
% 
% load iris_dataset
% X = irisInputs;
% N = size(irisInputs,2);
% 
% % set the fuzzifier constant to 1.4
% m = 1.4;
% swarm = RCE(X, 3, 'distance','mahalanobis','fuzzifier',m, 'display','text', ...
% 'swarm',6, 'subsprob',0.03, 'maxiter',100,'resampling_rate',0.8,'calculate_labels',false);
%
% [softlabels, labels] = swarm_cluster(X,swarm);
% 
% imagesc(cell2mat(softlabels'))
%
%

     softlabels = arrayfun(@(x) dist2memb(distmat(in, swarm.minima{x},swarm.distance,swarm.minkowski_p,swarm.Sigma_inv{x}),swarm.fuzzifier),1:length(swarm.minima),'uniformoutput',false);
     labels = cellfun(@(x) argmax(x,1),softlabels,'uniformoutput',false);
     crisplabels = cellfun(@(softlabels,labels) bsxfun(@eq,labels',1:size(softlabels,1))',softlabels,labels,'uniformoutput',false);
end


function I = argmax(x,dim)
    [~,I] = max(x,[],dim);
end

