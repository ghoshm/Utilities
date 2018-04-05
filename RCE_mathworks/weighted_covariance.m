function Sigma = weighted_covariance(in,U)
% calculate weighted covariance matrix
% Sigma is a dim x dim x K matrix of covariance for each k
% in is a dim x N matrix of input data
% U is a K x N matrix of fuzzy membership
    U = real(U);
    alpha_vec = sum(U,2);
    k = size(U,1);
    
    dim = size(in,1);
    Sigma = repmat(eye(dim),[1,1,k]);
    
    for i = 1:k
        if(alpha_vec(i) > 2)
            z_i = sum(bsxfun(@times,U(i,:),in),2)/alpha_vec(i);
            x_z_i = bsxfun(@minus,in,z_i);

            x_z_i = bsxfun(@times, U(i,:),x_z_i);

            Sigma_i = x_z_i*x_z_i'/sum(U(i,:));
            Sigma_i = 0.5*(Sigma_i' + Sigma_i );
             
            Sigma(:,:,i) = Sigma_i;
        else
            Sigma(:,:,i) = eye(dim);
        end
    end
end
