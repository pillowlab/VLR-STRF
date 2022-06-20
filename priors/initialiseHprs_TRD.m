function hprs =  initialiseHprs_TRD(k0,tmax)
% hprs =  initialiseHprs_TRD(k0,tmax)
%
% Initialize hyperperparameters of temporal relevance determination (TRD)
% model for temporal RF components
%
% INPUT:
%  k0 [nt x nx] - spike triggered average (time x space, unrolled as a matrix)
%  tmax [1 x 1] - length of temporal filter (in s)
%
% OUTPUT:
%  hprsTime - initial TRD hyperparameters, which are:
%
%     hprs.rho = marginal variance
%     hprs.len = length scale
%     hprs.c   = warping parameter

nt = size(k0,1); % number of time bins

% Take SVD of (temporally smoothed) initial full filter
[~,s] = svd(gsmooth(k0,2));

% Build struct
hprs.rho = s(1)^2/nt;  % marginal variance
hprs.len = tmax/3; % initialize length scale to 1/4 of range of filter
hprs.c = 0.5; % nonlinear stretch factor

