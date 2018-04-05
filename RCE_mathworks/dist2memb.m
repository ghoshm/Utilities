function [fmemb dtolamb] = dist2memb(d_mx,m,type)
%m = unifrnd(0.1,100);
if(nargin<3)
    type = 2;
end

switch type 
    case 1 %soft k-means
        dtolamb = exp(-bsxfun(@times, eps + d_mx,abs(m(:))));
        fmemb = bsxfun(@rdivide,dtolamb,eps+sum(dtolamb,1));
    case 2 %bezdek
        dtolamb = bsxfun(@power,eps + d_mx,(-1./(m(:)-1)));
        fmemb = bsxfun(@rdivide,dtolamb,eps+sum(dtolamb,1));
    case 3 %k-means
        [~,idx] = min(d_mx,[],1);
        fmemb = bsxfun(@eq,(1:size(d_mx,1))',idx);
end
