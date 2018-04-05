function M = distmat(in,x,distance, P,Sigma_inv)   
% M = distmat(in,x,distance_num,minkoski_p,Sigma_inv)
%     fully vectorized distance matrix computation
% in = d x N matrix of N input observations
% x = d x K matrix of K centroids
% distance = the distance switch as follows:
%           1: 'euclidean'    Eucledian
%           2: 'sqeuclidean'  Squared Euclidean
%           3: 'maxdiff'Manhattan
%           4: Minkowski Metric
%           5: Chebyshev Distance
%           6: Correlation
%           7: Cosine
%           8: Squared Correlation
%           9: Spearman's Rho
%           10: Mahalanobis distance
%           11: Log Spectral Distance (incomplete, don't use this)
%           12: Symmetrical Kullback - Leibler Divergence ---
%           13: Jensen - Shannon Distance (square root of JS divergence) ---
%               When using the divergence family, the input vectors must be
%               probability such that all elements in x are floating points between 
%               {0,1}. 
%           14: beta divergence
%           15: 1 - Jaccard similarity for categorical vectors
%           16: 1 - Sorensen-Dice Similarity for categorical vectors
%           17: Normalized Hamming distance
% P = the minkowski power
% Sigma_inv = d x d x K matrix of inverse of the covariance of each centroid

    switch distance
        case 1
            M = my_euc(x,in);     
        case 2
            M = my_sqeuc(x,in);
        case 3
            M = my_cityblk(x,in);
        case 4
            M = my_minkowski(x,in,P);
        case 5
            M = my_maxdiff(x, in);
        case 6
            M=my_corr(x,in,2);           
        case 7
            M=my_corr(x,in,1);           
        case 8
            M=my_corr(x,in,3);  
        case 9
            M = my_spearman(x,in);
        case 10
            M = my_mahal(x,in,Sigma_inv);     
        case 11
            M = my_LS(x,in);
        case 12
            M = KL(x,in);
        case 13
            M = JS(x,in);
        case 14
            M = beta(x,in);
        case 15
            M = my_jaccard(round(x),round(in));
        case 16
            M = my_sorensen(round(x),round(in));
        case 17
            M = my_hamming(round(x),round(in));
    end
end



function C = my_corr(A, B, type)
    ns1 = sqrt(sum(A .* A, 1));
    ns2 = sqrt(sum(B .*B, 1));
    ns1(ns1 == 0) = 1;  
    ns2(ns2 == 0) = 1;
    C = bsxfun(@times, A' * B, 1 ./ ns1');
    C = bsxfun(@times, C, 1 ./ ns2);
    switch type
        case 1
            C = 1 - C;
        case 2
            C = real(acos(C));
        case 3
            C = real(sqrt(1-C.^2));
    end
end

function C = my_sqeuc(A,B)
     C = abs(bsxfun(@plus,dot(A,A,1)',dot(B,B,1))-2*(A'*B));
end


function C = my_cityblk(A,B)
    C = zeros(size(A,2),size(B,2));
    for i = 1:size(A,2)
        C(i,:) = sum(abs(bsxfun(@minus,B,A(:,i))),1);
    end   
end

function C = my_euc(A,B)
     Eusq = abs(bsxfun(@plus,dot(A,A,1)',dot(B,B,1))-2*(A'*B));
     % Newton raphson approximation of sqrt
     C = sqrt(Eusq);
end


function C = my_minkowski(A,B,P)    
    C = zeros(size(A,2),size(B,2));
    for i = 1:size(A,2)
        C(i,:) = sum(abs(bsxfun(@minus,B,A(:,i))).^P,1).^(1/P);
    end
end

function C = my_maxdiff(A,B)    
    C = zeros(size(A,2),size(B,2));
    for i = 1:size(A,2)
        C(i,:) = max(abs(bsxfun(@minus,B,A(:,i))),[],1);
    end
end

function C = my_mahal(A,B,M)  
    %M = nancov(B')\eye(size(B,1));
                
    
    C = zeros(size(A,2),size(B,2));
    
    for i = 1:size(A,2)
        R = chol(M(:,:,i));
        RA = R*A(:,i);
        RB = R*B;
        C(i,:) = bsxfun(@plus,full(dot(RB,RB,1)),full(dot(RA,RA,1))')-full(2*(RA'*RB));
        %C(i,:) = max(0,C(i,:) - log(det(M(:,:,i))));
    end
    %C = sqrt(C);
end



function C = my_hamming(A,B)  
    C = zeros(size(A,2),size(B,2));
    for i = 1:size(A,2)
        C(i,:) = sum(bsxfun(@ne,A(:,i),B),1);
    end    
    C = C/size(A,1);
end

function C = my_LS(A,B)

    logB = log10(B);
    logA = log10(A);
    C = zeros(size(A,2),size(B,2));
    for i = 1:size(A,2)
        tmp = bsxfun(@minus,logB,logA(:,i));
        C(i,:) = sum(full(dot(tmp,tmp)),1);
    end
    %C = (C));
end


function C = beta(A,B)
    %A = max(min(A,1),0);
    %B = max(min(B,1),0);
    maxAB = max([A,B],[],2) ;
    A = exp(-bsxfun(@rdivide,A,maxAB));
    B = exp(-bsxfun(@rdivide,B,maxAB));
    
    
    %A = bsxfun(@rdivide,A, mean(A,1));
    %B = bsxfun(@rdivide,B, mean(B,1));
    logA = log2(A + eps);
    logB = log2(B + eps);
    AlogA = dot(A,logA)';
    BlogB = dot(B,logB);
    C = bsxfun(@plus,-A'*logB,AlogA) + bsxfun(@plus,-(B'*logA)',BlogB);
    C = C.*my_cityblk(A,B);
    %C = C./size(A,1);
end


function C = KL(A,B)


    logA = log2(A + eps);
    logB = log2(B + eps);
    AlogA = dot(A,logA)';
    BlogB = dot(B,logB);
    C = abs(bsxfun(@plus,-A'*logB,AlogA) + bsxfun(@plus,-(B'*logA)',BlogB));
    %C = C./size(A,1);
end



function C = JS(A,B)

    logA = log2(A + eps);
    logB = log2(B + eps);
    AlogA = dot(A,logA)';
    BlogB = dot(B,logB);
    
    C = zeros(size(A,2),size(B,2));
    for i = 1:size(A,2)
%         i
        M = bsxfun(@plus,A(:,i),B)/2;
        logM = log2(M + eps);
        C(i,:) = bsxfun(@plus,-A(:,i)'*logM,AlogA(i)) -(sum(B.*logM,1)) + BlogB;
    end
    C = sqrt(C);%./size(A,1);
end

function C = my_spearman(A,B)
    n = size(A,1);
    rankA = tiedrank(A,1);
    rankB = tiedrank(B,1);
    dsq = abs(bsxfun(@plus,dot(rankA,rankA,1)',dot(rankB,rankB,1))-2*(rankA'*rankB)); %sq euclidean
    C = 6*dsq./(n*(n*n-1));
end

function C = my_jaccard(A,B)
    C = zeros(size(A,2),size(B,2));
    for i = 1:size(A,2)
        AandB = bsxfun(@and, A(:,i),B).*bsxfun(@eq, A(:,i),B);
        AorB = bsxfun(@or, A(:,i),B);
        C(i,:) = sum(AandB,1)./sum(AorB,1);
    end
    C = 1 - C;
end


function C = my_sorensen(A,B)

    C = zeros(size(A,2),size(B,2));
    d = size(A,1);
    for i = 1:size(A,2)
        AandB = bsxfun(@and, A(:,i),B).*bsxfun(@eq, A(:,i),B);
        C(i,:) = sum(AandB,1);
    end
    C = 1 - C/size(A,1);
end

