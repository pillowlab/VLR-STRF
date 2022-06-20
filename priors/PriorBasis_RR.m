function [Bkx, dBkx, BasisInfo] = PriorBasis_RR(hprs,Prior,BasisInfo, varargin);
% [Bkx,dBkx] = PriorBasis_RR(hprs,Prior)
%
% Compute basis for RR prior covariance matrix (and its derivative), such
% that samples from the prior can be computed via:
%
%       w_x = Bkx*eps,  where eps ~ N(0, I) is a standard normal.
% 
% Input
% -----
%       hprs - vector of hyperparameters  [scale]
%       Prior - struct with basis information
%
% Output
% ------
%       Btx [nkt, m] -  basis
%   dBkx [nkt, m, 3] - derivative of basis w.r.t hyperparameters

if length(hprs) > 1
    error('too many hyperparameter inputs')
end

dims = Prior.dims;

if nargin < 3
    BasisInfo = [];
end
    
if isempty(BasisInfo)
    BasisInfo.idx = true(prod(dims),1); % place holder to make output consistent with other function calls
end

if length(hprs) == 1
    ridge = hprs;
else
    ridge = 1;
end

Bkx = sqrt(ridge) * eye(prod(dims));    
dBkx = 0.5./sqrt(ridge) * eye(prod(dims));

