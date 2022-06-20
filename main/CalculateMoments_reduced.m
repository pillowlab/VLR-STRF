function [XY,Xmu,XXr,Ymu,YY] = CalculateMoments_reduced(Resp,Stim,nkt)
% [XY,Xmu,XXr,Ymu,YY] = CalculateMoments_reduced(Resp,Stim,nkt)
%
% Computes mean and reduced representation of spike-triggered stimuli and
% raw stimuli 
%
% INPUT:
%    Resp [n x 1] - response vector (n time bins)
%    Stim [n x m] - stimulus matrix (n time bins x m space dims)
%     nkt [1 x 1] - # time bins to include as part of stimulus
% MaxSize [1 x 1] - max # of floats to store while computing cov
%                      (smaller = slower, but less memory needed)
%                      (optional)  [Default: 1e9]
%
%  OUTPUT:
%    XY [ nkt   x   m ]  - response-triggered stimulus (STA) X'Y 
%   Xmu [ nkt   x   m ]  - raw sum of X        (1st stim moment)
%    XX [ m x m x nkt ]  - subset of raw sum of X'*X (2nd stim moments)
%   Ymu [ 1     x   1 ]  - mean spike response
%    YY [ 1     x   1 ]  - second moment of spike response


% Get sizes
[T,nkx] = size(Stim); % stimulus size (# time bins x # spatial bins).

% Allocate space
XY = zeros(nkt,nkx);
Xmu = zeros(nkt,nkx);
XXr = zeros(nkx,nkx,nkt);

% Initialize
XY(end,:) = Resp'*Stim;
Xmu(end,:) = sum(Stim);
XXr(:,:,end) = Stim'*Stim;

% Compute
for jj = 2:nkt
    
    % 1st moments
    XY(nkt-jj+1,:) = Resp(jj:end)'*Stim(1:end-jj+1,:);
    Xmu(nkt-jj+1,:) = Xmu(nkt-jj+2,:)-Stim(end-jj+2,:);
    
    % 2nd moments
    XXr(:,:,nkt-jj+1) = Stim(jj:end,:)'*Stim(1:end-jj+1,:);

end

Ymu = sum(Resp);
YY = Resp'*Resp;

% Compute weights (to correct moments): currently not used
% wts = bsxfun(@minus,(T-nkt+1:T)',0:nkt-1);
% wts = bsxfun(@rdivide, wts, T:-1:T-nkt+1);
% wts = (T-nkt+1:T)'/T;
