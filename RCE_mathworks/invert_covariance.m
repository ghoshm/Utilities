function M_inv = invert_covariance(M)
% M inverts the matrices in the cells M
% if M is a dim x dim x K double matrix, then M_inv is simply inverts M
% usage:
% Sigma = invert_covariance(swarm.Sigma_inv)
%

    invert_matrices = @(M) cell2mat(reshape(arrayfun(@(i) mldivide(M(:,:,i),eye(size(M,1))),1:size(M,3),'uniformoutput',false),1,1,[]));
    if(iscell(M))
        M_inv = cellfun(invert_matrices,M,'uniformoutput',false);
    else
        M_inv = invert_matrices(M);
    end
end