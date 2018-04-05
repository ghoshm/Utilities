function [h,x,y,z] =plot_fuzzy_voronoi (dt, u, m,N_steps,distance,P,M)
% plot the 2-dimensional fuzzy voronoi cell for fuzzy clustering
% dt = a 2 \times N matrix of observations
% u = a 2 \times K matrix of centroids
% m = a floating point m \in \{ 1, 2 \} the fuzzifier (see bezdek's formula for fcm)
% N_steps = number of steps for the contour, higher makes the plot smoother
% distance = distmat distance id; check distmat documentation
% P = minkowski P
% M = inverse of the covariance matrix
% f_num = figure number


if(nargin < 5)
    distance = 1;
end

%clf;
D = distmat(dt,u,distance,P,M);

[~,memb] = dist2memb(D,m);
if(~ishandle(1))
    figure(1)
    clf
    scatter(dt(1,:),dt(2,:), 10, max(log(memb),[],1), 'o','filled');
else
    figure(1)
end
hold on;
%v = voronoi(u(1,:),u(2,:));


int_x = linspace(min(get(gca,'xlim')), max(get(gca,'xlim')),N_steps);
int_y = linspace(min(get(gca,'ylim')), max(get(gca,'ylim')),N_steps);

[x,y] = meshgrid(int_x,int_y);
in = [x(:),y(:)]';
%in = repmat(x(:)',size(dt,1),1);
D = distmat(in,u,distance,P,M);

[~,memb] = dist2memb(D,m);

z(:,:,1) = reshape(max(log(memb),[],1),size(x));

hold on;
[C,b] = contour(x,y,z(:,:,1));
%clabel(C,b)
%set(v,'linewidth',0.5,'color','b');
scatter(u(1,:),u(2,:),150,'kx','linewidth',2);

h = gca;


%clf;
D = distmat(dt,u,distance,P,M);

fm = dist2memb(D,m);

if(~ishandle(2))
    figure(2)
    clf
    scatter(dt(1,:),dt(2,:), 10, max(fm.^m,[],1), 'o','filled');
else
    figure(2)
end
hold on;
%v = voronoi(u(1,:),u(2,:));


int_x = linspace(min(get(gca,'xlim')), max(get(gca,'xlim')),N_steps);
int_y = linspace(min(get(gca,'ylim')), max(get(gca,'ylim')),N_steps);

[x,y] = meshgrid(int_x,int_y);
in = [x(:),y(:)]';

D = distmat(in,u,distance,P,M);

fm = dist2memb(D,m);

z(:,:,2) = reshape(max(fm,[],1),size(x));

hold on;
[C,b] = contour(x,y,z(:,:,2));
%clabel(C,b)
%set(v,'linewidth',0.5,'color','b');
scatter(u(1,:),u(2,:),150,'kx','linewidth',2);

h = gca;