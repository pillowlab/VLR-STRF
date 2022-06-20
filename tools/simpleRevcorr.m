function [XY,XX] = simpleRevcorr(X,Y,nt)
% [XY,XX] = simpleRevcorr(X,Y,nkt)
%
% Computes reverse correlation of Y with X, for a number of lags nkt.
% Optionally computes autocovariance of X (expensive if X has many columns)
%
% INPUT:
%        X [N x M] - stimulus matrix; 1st dimension = time, 2nd dimension = space
%        Y [N x 1] - column vector of spike count in each time bin (can be sparse)
%       nt [1 x 1] - # time samples to consider to be part of the stimulus
%
% OUTPUT:
%    XY [nt x M]      - reverse correlation of Y with X (spike-triggered sum, reshaped as matrix)
%    XX [nt*M x nt*M] - autocovariance of X with itself, rearranged as matrix (OPTIONAL)
%
% Note: this implementation is not 

swid = size(X,2); % stimulus size (# time bins x # spatial bins).

% Compute reverse correlation of X against Y
XY = zeros(nt,swid); % allocate space
for j = 1:nt
    XY(j,:) = (Y(nt-j+1:end)'*X(1:end-nt+j,:));
end

% Compute stimulus autocovariance, if desired
if nargout > 1
    xc = xcorr(X,nt-1); % raw cross-correlation
    XX = zeros(nt,nt,swid^2);  % allocate space for XX
    for jj = 1:swid^2  % make toeplitz matrices for each block
        XX(:,:,jj) = toeplitz(xc(nt:end,jj),flipud(xc(1:nt,jj)));
    end
    % reshape it into covariance matrix X'*X;
    XX = squeeze(reshape(permute(reshape(XX,nt,nt*swid,swid),[1 3 2]),nt*swid,1,nt*swid));
end

