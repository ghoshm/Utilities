function swarm = RCE(in,k, varargin )
%RCE Rapid Centroid Estimation (RCE) clustering (2014)
%   swarm = RCE(X, K) partitions the points in the Dim-by-N data matrix X
%   into K clusters.  This partition minimizes the sum, over all clusters, of
%   the within-cluster sums of point-to-cluster-centroid distances.
%   
%   Rows of X correspond to features, columns correspond to variables.  
%   By default, RCE uses Euclidean distances.
%
%   swarm = RCE(X, K) returns the summary statistics of the search
%
%   [ ... ] = RCE(..., 'PARAM1',val1, 'PARAM2',val2, ...) specifies
%   optional parameter name/value pairs to control the iterative algorithm
%   used by RCE.  Parameters are:
%
%   'Distance' - Distance measure, in P-dimensional space, that RCE
%      should minimize with respect to.  Choices are:
%          'euclidean', 'sqeuclidean', 'cityblock', 'minkowski',
%          'chebyshev','cosine','correlation','sqcorrelation','spearman',
%          'normalized_mahalanobis','log-spectral',
%          'KL-divergence','JS-divergence',
%          'beta', 'jaccard', 'sorensen-dice','hamming'
%         
%      Note that these distances are based on the 
%      distmat(in,x,distance_num,minkoski_p,Sigma_inverse)
%      fully vectorized distance matrix computation:
%         1: 'euclidean'    Eucledian
%         2: 'sqeuclidean'  Squared Euclidean
%         3: 'maxdiff'Manhattan
%         4: Minkowski Metric
%         5: Chebyshev Distance
%         6: Correlation
%         7: Cosine
%         8: Squared Correlation
%         9: Spearman's Rho
%         10: Mahalanobis distance --- 
%             uses covariance matrix, RCE will automatically optimize the
%             seed and covariances using beta divergence first.
%         11: Log Spectral Distance (incomplete, don't use this)
%         12: Symmetrical Kullback - Leibler Divergence ---
%         13: Jensen - Shannon Distance (square root of JS divergence) ---
%             When using the divergence family, the input vectors must be
%             probability such that all elements in x are floating points between 
%             {0,1}. 
%         14: beta divergence
%         15: 1 - Jaccard similarity for categorical vectors
%         16: 1 - Sorensen-Dice Similarity for categorical vectors
%         17: Normalized Hamming distance
%
%   Other Parameters:
%          'P'        - Set the P value for minkowski distance, default is 2
%          'subsprob' - Set the substitution probability (for RCE+),
%          default is 0.05
%          'swarm'    - Set the number of groups in the swarm (for swarm
%                       RCE), default is 1 (normal RCE)
%          'MaxStagnation' - Set the maximum iteration to stagnate before 
%                       particle reset (for RCE^r), default is 15
%          'fun'      - defines the minimizing fitness function. The RCE 
%                       abstract fitness function should obey the syntax of
%                       fit = function_name(distance_matrix,labels)
%                       the default fitness function used in RCE is the sum 
%                       of intracluster distances normalized by data volume.
%          'Display'  - Level of display output.  Choices are 'text', (the
%                       default), 'off', and 'on'.
%          'MaxIter'  - Maximum number of iterations allowed.  Default is 100.
%          'Evolution' - Allows the RCE to summon particles at will in
%                        order to satisfy the cluster entropy criterion.
%                        Cluster entropy depends on 'Entropy_Target' and
%                        'fuzzifier'
%          'Entropy_Target' - A vector of target entropies for each
%                             subswarm, if 'Evolution' is true then each
%                             subswarm will summon additional particles to
%                             satisfy the target entropy
%          'fuzzifier'        - The fuzzifier parameter as per Bezdek's Fuzzy C-means
%          'resampling_rate'  - a parameter from 0 to 1, denoting the rate of
%                       random sample for each subswarm
%          'calculate_labels' - If true calculate the crisp and fuzzy labels for 
%                               each subswarm. Setting this to False (default) prevents
%                               RCE from calculating the labels for the whole 
%                               range of input vectors to conserve memory.
%          'calculate_covariance' - If set to true, RCE will calculate the 
%                                   covariance matrix regardless of the distance 
%                                   function used, if distance is
%                                   set to 'mahalanobis', then covariance
%                                   matrix is automatically calculated. (The
%                                   default setting is False)
%
%
%   Example:
%     % An example run of the algorithm using Iris Dataset
% 
%     clear all
%     close all;
% 
%     % load iris dataset
%     load iris_dataset
%     X = irisInputs;
%     N = size(irisInputs,2);
% 
%     % set the fuzzifier constant to 1.4
%     m = 1.4;
% 
%     % Optimize the swarm using 80% resampling rate and mahalanobis distance
%     swarm = RCE(X, 3, 'distance','mahalanobis','fuzzifier',m, 'display','text', ...
%     'swarm',6, 'subsprob',0.03, 'maxiter',100,'resampling_rate',0.8,'calculate_labels', false);
% 
%     % calculate the fuzzy labels, crisp labels, and numeric labels from the
%     % input vectors using the Swarm
%     [softlabels, crisplabels, numlabels] = swarm_cluster(X,swarm);
% 
%     % plot the fuzzy voronoi cells on the 1st and 3rd dimension
%     visualize_swarm(X,swarm,1,3,m,200)
% 
%     % Perform fuzzy evidence accumulation on the swarm
%     ensemble = EnsembleAggregate(softlabels,'average',true);
% 
%     % plot the scatter matrix             
%     figure('name','scatterplot');
%     gplotmatrix(X(1:4,:)',[],ensemble.ensemble_labels)
%
%   See also KMEANS, LINKAGE, CLUSTERDATA, SILHOUETTE.
%
%   RCE is proposed as a derivate of PSC algorithm with radically reduced 
%   time complexity. The quality of the results produced are similar to PSC.
%   The advantage of RCE is RCE has lesser complexity and
%   faster convergence [1-2]. Recent updates to RCE removes the
%   redundancies of RCE and further decrease the overall complexity, 
%   enabling it to perform Ensemble Clustering in quasilinear complexity.
%   The additional algorithms are Tsaipei-Wang's CA-tree and Fuzzy Evidence
%   Accumulation. The code will be provided seperately in the next update.
%
%   In 2014 the main components in RCE are reduced to only 2
%   1. Self-organizing : Each data points in the cluster tries to pull the
%                        corresponding RCE particle to cluster centre of mass
%   2. Swarm Best : The optimum solution discovered by the swarm
%
%   Substitution strategy allows RCE particle to distrupt an achieved
%   equilibrium position by moving towards other particles. [1]
%
%   Particle reset reinitialize the x and dx of every particle in the group
%   when stagnation counter exceeds MaxStagnation [1]   
%
%   Swarm strategy allows multiple subswarms of RCE to work in paralel. 
%   In the swarm strategy each subswarm will share its knowledge to other 
%   subswarms. [1]
%
%   Evolution strategy is a reinterpretation of the charged particles
%   strategy in RCE so that the number of particles can be increased
%   automatically. This strategy works best in Gaussian data. For
%   non-convex data, it is better to initialize rce with fixed amount of K
%   and perturbing the data, e.g. nsample = 0.9.
%
%   An increase of computational complexity is apparent in the current swarm 
%   strategy. One should consider the computational load that may result from adding 
%   more subswarms.
%
% References:
%   [1] M. Yuwono, S.W. Su, B. Moulton, & H. Nguyen, "Data Clustering Using 
%       Variants of Rapid Centroid Estimation", IEEE Transactions on
%       Evolutionary Computation, Vol 18, no.3, pp.366-377. ISSN:1089-778X.
%       DOI:10.1109/TEVC.2013.2281545.
%   [2] M. Yuwono, S.W. Su, B. Moulton, H. Nguyen, "An Algorithm for Scalable
%       Clustering: Ensemble Rapid Centroid Estimation", in Proc 2014 IEEE 
%       Congress on Evolutionary Computation, 2014, pp.1250-1257. 
%
%   Copyright 2011-2014 Mitchell Yuwono.
%   $Revision: 1.1.0 $  $Date: 2014/10/21 15:19:00 $
    

    % parse input arguments
    
    pnames = {'distance' 'maxiter' 'display' 'subsprob' 'P' 'swarm' 'fun' 'maxstagnation' 'resampling_rate' 'entropy_target' 'evolution' 'fuzzifier' 'calculate_labels' 'calculate_covariance'};
    dflts =  {'sqeuclidean' 100      'text' 0.03 2 3 'wcdist' 15 1 [] false 2 false false};
    
    parser = inputParser;
    for i = 1:length(pnames)
        parser.addParamValue(lower(pnames{i}),lower(dflts{i}));
    end
    parser.parse(varargin{:});
    r = parser.Results;
    
    
    distNames = {'euclidean', 'sqeuclidean', 'cityblock', 'minkowski', 'chebyshev','cosine','correlation','sqcorrelation','spearman','mahalanobis','log-spectral','KL-divergence','JS-divergence','beta', 'jaccard', 'sorensen-dice','hamming'};
    funNames = {'wcdist', 'sildist','dunndist','sildistwc'};
    dispNames = {'off','on','text'};
    validatestring(r.distance,distNames);
    validatestring(r.fun,funNames);
    validatestring(r.display,dispNames);
    
    calc_covar = r.calculate_covariance;
    maxiter = r.maxiter;
    display = find(strcmp(r.display,dispNames) == 1);
    max_stagnation = r.maxstagnation;
    substitution_probability = r.subsprob;
    evolution = r.evolution;
    nsample = r.resampling_rate;
    nm = r.swarm;
    m = r.fuzzifier;
    fun = find(strcmp(r.fun,funNames) == 1);
    distance = find(strcmp(r.distance,distNames) == 1);
    minkowksi_p = r.p;
    
    fit_th = r.entropy_target;
    if(isempty(fit_th))
        fit_th = linspace(0.01,0.25,nm);
    end
    
    
    switch fun
        case 1
            f = @(D_mat,I_idx) wcdist(D_mat,I_idx);
        case 2
            f = @(D_mat,I_idx) sildist(D_mat,I_idx);
        case 3
            f = @(D_mat,I_idx)dunndist(D_mat,I_idx);
        case 4 
            f = @(D_mat,I_idx) sildist(D_mat,I_idx) + wcdist(D_mat,I_idx) + dunndist(D_mat,I_idx);                            
    end
    
    %% basic memory
    disp('initializing');
    
    
    disp('Running RCE');
    
    tic
    if(distance == 10)
        disp('optimizing initial centroids...');
        [~, M,~,~,~,~,Sigma_inv] = rceoptimize(in,k,m,2*maxiter,0,max_stagnation,substitution_probability,0.95*nsample,nm,f,14,minkowksi_p, fit_th,evolution,[],true);
        seed.x = M;
        seed.Sigma_inv = Sigma_inv;
        disp('finalizing covariances');
        [fit, M, Im, numclust, ni, idx_in,Sigma_inv, it_add] = rceoptimize(in,k,m,maxiter,display,max_stagnation,substitution_probability,nsample,nm,f,distance,minkowksi_p, fit_th,false,seed,calc_covar);
    else
        [fit, M, Im, numclust, ni, idx_in,Sigma_inv, it_add] = rceoptimize(in,k,m,maxiter,display,max_stagnation,substitution_probability,nsample,nm,f,distance,minkowksi_p, fit_th,evolution,[],calc_covar);
    end
    t = toc;
    %% format output    
    if(r.calculate_labels)
        swarm.softlabels = arrayfun(@(x) dist2memb(distmat(in,M{x},distance,minkowksi_p,Sigma_inv{x}),m),1:length(M),'uniformoutput',false);
        swarm.labels = cellfun(@(x) argmax(x,1),swarm.softlabels,'uniformoutput',false);
    end
    
    swarm.fitness_graph = fit;
    
    swarm.minima = M;
    
    if(nm > 1)
        swarm.swarm_minimum_centroids = M{Im};
        swarm.swarm_fitness = min(fit.average_distortion,[],1);
    end
    
    swarm.numclust = uint32(numclust);
    
    swarm.ni = ni;
    swarm.rsample = idx_in;
    swarm.t = t;
    
    swarm.Sigma_inv = Sigma_inv;
    swarm.minkowski_p = minkowksi_p;
    swarm.distance = distance;
    swarm.fuzzifier = m;
    swarm.n_particles = uint32(it_add); 
    clearvars -except swarm in;
    
    dt = whos;
    swarm.memory = dt;
    
end

function I = argmin(x,dim)
    [~,I] = min(x,[],dim);
end

function I = argmax(x,dim)
    [~,I] = max(x,[],dim);
end


function fit = wcdist(distance_matrix,label)
   if(~isempty(distance_matrix))
        tmp = distance_matrix;
        k = 1:size(distance_matrix,1);
        U = bsxfun(@eq,label,k');
        fit = sum(mean(U.*tmp,2));
   else
       fit = inf;
   end

    
end

function [fit, DN] = dunndist(distance_matrix, label)
    distance_matrix(:,label==0) = [];
    label(label ==0) = [];
    [a,~,label] = unique(label);

    k = numel(a);
    d = distance_matrix(a,:);
    
    delta = zeros(k,k);
    Delta = zeros(1,k);
    
    id = zeros(size(d)) == 1;
    for i = 1:k
        id(i,:) = ismembc2(label,i);
    end
    
    id = id == 1;
    
    for i = 1:k;
        idx = id(i,:);
        nS = nnz(idx);
        for j = 1:k;
            if(i ~= j)
               idy = id(j,:);
               nT = nnz(idx);
               delta(i,j) = (sum(d(i,idy)) + sum(d(j,idx)))/(nS+nT);             
            end            
        end                
        Delta(i) = 2*sum(d(i,idx))/nS;
    end
    delta = delta(:);
    delta = delta(delta>0);
    DN = min(delta(:))/max(Delta(:));
    if(k == 1)
        DN = 0;
    end
    
    fit = 1/DN;
    
    if(isempty(fit))
        fit = inf;
    end
    
end


function [fit, indv] = sildist(distance_matrix,label)
    
    [k,n] = size(distance_matrix);
    
    a = zeros(1,n);
    tmp = distance_matrix;
    
    
    for i = 1:k;
        g = logical(ismembc2(label,i));
        
        a(g) = tmp(i,g);        
        tmp(i,g) = inf;
    end
    b = min(tmp);

    
    indv = (b-a)./max(a,b);

    fit = mean(bsxfun(@minus,1,indv));
    
end


function [stats, M, Im, numclust, ni, idx_in,Sigma_inv, k_particles] = rceoptimize(in,K,m_f,maxiter,display,max_stagnation,substitution_probability,nsample,nm,fun,distance,P, fit_th,evolution,seed,calculate_covariance)
%% RCE Operations October 2014 Version 
% don't use this as is, refer to RCElite wrapper
% 
% K = 15;
% nm = 4;
% m_f = 2;
% distance = 2;
% display = 2;
% nsample = 0.8;
% P = 1;
% fun = 1;
% fit_th = linspace(0.1,0.2,nm);
% evolution = false;
% maxiter = 200;
% substitution_probability = 0.05;
% max_stagnation = 15;



    nsample = nsample*ones(1,nm);
    fuzzifier = num2cell(m_f*ones(1,nm));

    if(isempty(seed))
        if(evolution)
            k_particles = randi([2,K],1,nm);
        else
            k_particles = K*ones(1,nm);
        end
    else
        k_particles = cellfun(@(M) size(M,2),seed.x);
    end
    % fuzzifier = unifrnd(1.4,m_f,1,nm);



    w = (exp(-(1:maxiter)/(0.1*maxiter)));

    f = fun;

    rsample = @(nsample) randsample(size(in,2),floor(nsample*size(in,2)));

    initialize_particles = @(k) unifrnd(min(in,[],2)*ones(1,k),max(in,[],2)*ones(1,k));
    initialize_velocities = @(k) zeros(size(in,1),k);
    initialize_sigma = @(k) repmat(eye(size(in,1)),[1,1,k]);

    calculate_membership = @(D_mat,m) dist2memb(D_mat,m,2);
    calculate_crisp_membership = @(D_mat) dist2memb(D_mat,[],3);
    Psi = @(in,x,u) bsxfun(@rdivide,in*u',eps+sum(u,2)') - x;

    stagnated = zeros(1,nm);

    numclust = uint16(zeros(1,nm));
    entrop = zeros(1,nm);

    fit = inf(nm,1);

    Ym = inf;

    if(display == 2)
        M_labels = uint16(zeros(nm,size(in,2)));
    end

    stagnation_counter = zeros(nm,1);
    if(isempty(seed))
        x = arrayfun(initialize_particles,k_particles,'uniformoutput',false);
    else
        x = seed.x;
    end
    
    M = x;
    v = arrayfun(initialize_velocities,k_particles,'uniformoutput',false);
    idx_in = arrayfun(rsample,nsample,'uniformoutput',false);
    ni = arrayfun(@(k) zeros(1,k),k_particles,'uniformoutput',false);

    if(isempty(seed))
        if(distance == 10)
            Sigma_inv = arrayfun(initialize_sigma,k_particles,'uniformoutput',false);
        else
            Sigma_inv = arrayfun(@(k) false,k_particles,'uniformoutput',false);
        end
    else
        Sigma_inv = seed.Sigma_inv;
    end

    for it = 1:maxiter

        if(all(stagnated>3))
            disp('Long stagnation detected... stopping.');
            break;
        end

        for m = 1:nm
            additional_particles = k_particles(m)-size(x{m},2);
            if(additional_particles > 0)
                x{m} = cat(2,x{m},initialize_particles(additional_particles));
                v{m} = cat(2,v{m},initialize_velocities(additional_particles));
                if(distance == 10)
                    Sigma_inv{m} = cat(3,Sigma_inv{m},initialize_sigma(additional_particles));
                end
            end

            D_mat = distmat(in(:,idx_in{m}),x{m},distance,P,Sigma_inv{m});

            if(distance == 10)
                U = calculate_membership(D_mat,fuzzifier{m});
                Sigma_tmp = weighted_invert_covariance(in(:,idx_in{m}),U.^fuzzifier{m});
            end


            % update minimum              

            [~,I_idx] = min(D_mat,[],1);
            fit_t = f(D_mat,I_idx);

            ni_m = arrayfun(@(i) nnz(I_idx == i),1:size(x{m},2));
            
            if(fit_t < fit(m,max(it-1,1)))     
                stagnated(m) = 0;
                ni{m} = ni_m; 
                M{m} = x{m}(:,ni{m}>0);
                
                numclust(m) = numel(unique(I_idx));


                stagnation_counter(m) = 0;

                if(distance == 10)
                    Sigma_inv{m} = Sigma_tmp;
                end

                if(distance ~= 10); U = calculate_membership(D_mat,fuzzifier{m}); end
                U = U(ni_m>0,:);
                entrop(m) = -mean(U(:).*log2(eps+U(:)));
                
                % check entropy, if entropy is low, increase k
                if(evolution && numclust(m) == k_particles(m))
%                    if(distance ~= 10); U = calculate_membership(D_mat,fuzzifier(m)); end
                   if( -mean(U(:).*log2(eps+U(:))) > fit_th(m)); k_particles(m) = k_particles(m) + randi(2); end
%                    if( sildist(D_mat,I_idx) > fit_th(m) ); k_particles(m) = k_particles(m) + randi(5);  end
                end

                % for displays
                if(display == 2)
                    %------------------------------
                    % for display, comment otherwise
                    labels_tmp = zeros(1,size(in,2));
                    labels_tmp(idx_in{m}) = I_idx;

                    M_labels(m,:) = labels_tmp;
                    %------------------------------
                end
            else
                fit_t = fit(m,max(it-1,1),1);
                stagnation_counter(m) = stagnation_counter(m) + 1;
            end

            fit(m,it) = fit_t;

            % redirect empty particles to winning particles
            [~,I_win] = max(ni_m);
            retract = zeros(size(x{m}));
            tmp = bsxfun(@minus,x{m}(:,I_win),x{m}(:,ni_m==0));
            retract(:,ni_m==0) = tmp;

            % calculate the self organizing vector;

            so = Psi(in(:,idx_in{m}),x{m},calculate_crisp_membership(D_mat));

            Dm = distmat(cell2mat(M),x{m},distance,P,Sigma_inv{m});
            mi = Psi(cell2mat(M),x{m},calculate_crisp_membership(Dm));

            v{m} = w(it)*v{m} + unifrnd(0,1,size(in,1),size(x{m},2)).*(so + mi + retract);
            x{m} = x{m}+v{m};

            % substitution
            Is = rand([1 size(x{m},2)]) < substitution_probability;

            x{m}(:,Is==1) = bsxfun(@plus, x{m}(:,I_win), 0.01*initialize_particles(nnz(Is)));
            v{m}(:,Is==1) = initialize_velocities(nnz(Is));
            
            

            % particle reset

            if(max_stagnation > 0 && stagnation_counter(m) > max_stagnation)                          
    %            disp('reset');
               stagnated(m) = stagnated(m) + 1;
               stagnation_counter(m) = 0;
               x{m} = initialize_particles(size(x{m},2));
               v{m} = initialize_velocities(size(x{m},2));
            end

        end

        if(display == 3)
            Ym = min(fit(:,it),[],1);
            disp(['[iter=' num2str(it, '%i') ']' ' fitness=' num2str(Ym) '   k=' mat2str(k_particles)]);
        end
        if(display == 2 && min(fit(:,it),[],1) < Ym)
            imagesc(M_labels); 
            pause(0.01);
        end
%         w(it)
%         pause(0.01)

        Km_graph(:,it) = numclust(:);
        Ent_graph(:,it) = entrop(:);
    end
    
    
    % calculate the inverse covariance matrix if calculate_covariance switch is on
    if(calculate_covariance)
        D_mat = cellfun(@(M,Sigma_inv,idx_in) distmat(in(:,idx_in),M,distance,P,Sigma_inv), M,Sigma_inv,idx_in,'uniformoutput',false);
        U = cellfun(@(D_mat,m) calculate_membership(D_mat,m), D_mat,fuzzifier,'uniformoutput',false);
        Sigma_inv = cellfun(@(U,idx_in,m) weighted_invert_covariance(in(:,idx_in),U.^m), U,idx_in,fuzzifier,'uniformoutput',false);
    end
    
    [~,Im] = min(fit(:,end),[],1);
    
    stats.average_distortion = fit;
    stats.number_of_clusters = Km_graph;
    stats.cluster_entropy = Ent_graph;
    
    
    
    
    % calculate_distance(x{1},Sigma_inv{1});
end

function Sigma_inv = weighted_invert_covariance(in,U)
% calculate weighted covariance matrix
% Sigma is a dim x dim x K matrix of covariance for each k
% in is a dim x N matrix of input data
% U is a K x N matrix of fuzzy membership
    U = real(U);
    alpha_vec = sum(U,2);
    k = size(U,1);
    
    dim = size(in,1);
    Sigma_inv = repmat(eye(dim),[1,1,k]);
    
    for i = 1:k
        if(alpha_vec(i) > 2)
            z_i = sum(bsxfun(@times,U(i,:),in),2)/alpha_vec(i);
            x_z_i = bsxfun(@minus,in,z_i);

            x_z_i = bsxfun(@times, U(i,:),x_z_i);

            Sigma_i = x_z_i*x_z_i'/sum(U(i,:));
            Sigma_i = 0.5*(Sigma_i' + Sigma_i );
             
            Sigma_inv(:,:,i) = Sigma_i \ eye(dim);
        else
            Sigma_inv(:,:,i) = eye(dim);
        end
    end
end
