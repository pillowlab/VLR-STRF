function [XY,Xmu,XX,Ymu,YY] = CalculateMoments_full(Resp,Stim,nkt,maxsize)
% [XY,Xmu,XX,Ymu,YY] = CalculateMoments_Full(Resp,Stim,nkt,maxsize)
%
% Computes mean and covariance of spike-triggered (or response weighted)
% stimuli and raw stimuli
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
%    XY [ nkt   x   m   ]  - response-triggered stimulus (STA) X'Y 
%   Xmu [ nkt   x   m   ]  - raw sum of X        (1st stim moment)
%    XX [ nkt x nkt x m x m ]  - raw sum of X'*X (2nd stim moments)
%   Ymu [ 1     x   1 ]  - mean spike response
%    YY [ 1     x   1 ]  - second moment of spike response
%
%  Notes:  
%  (1) Zero-pads stimuli at beginning
%  (2) Reduce 'maxsize' if getting "out of memory" errors
%
% Dependencies: makeStimRows.m


%-------- Parse inputs  ---------------------------
if nargin < 4
    maxsize = 1e9; % max chunk size; decrease if getting "out of memory"
end

[T,nkx] = size(Stim); % stimulus size (# time bins x # spatial bins).
Msz = T*nkx*nkt;   % Size of full stimulus matrix in # floats

if Msz <= maxsize  % Check if stimulus is small enough to do in one chunk

    Xdesign = makeStimRows(Stim,nkt,'same'); % Make design matrix from full stimulus
    XX = Xdesign'*Xdesign;
    XY = Xdesign'*Resp;
    Xmu = sum(Xdesign)';
    
else  % Compute Full Stim matrix in chunks, compute mean and cov on chunks
    
    nchunk = ceil(Msz/maxsize);
    chunksize = ceil(T/nchunk);
    fprintf(1, 'calculateMoments_Full: using %d chunks to compute covariance\n', nchunk);
    
    % Compute statistics on first chunk
    imax = min(T-nkt+1,chunksize);  % ending index for chunk
    Xdesign = makeStimRows(Stim(1:imax+nkt-1,:),nkt,'same'); 
    XX = Xdesign'*Xdesign;
    XY = Xdesign'*Resp(1:imax+nkt-1);
    Xmu = sum(Xdesign)';
    
    % Compute for remaining chunks
    for j = 2:nchunk
        i0 = chunksize*(j-1)+1;  % starting index for chunk
        imax = min(T-nkt+1,chunksize*j);  % ending index for chunk
        Xdesign = makeStimRows(Stim(i0:imax+nkt-1,:),nkt,'valid');
        XX = XX + Xdesign'*Xdesign;
        XY = XY + Xdesign'*Resp(i0:imax);
        Xmu = Xmu+sum(Xdesign)';
    end

end

% Reshape stuff
XY = reshape(XY,nkt,nkx);
Xmu = reshape(Xmu,nkt,nkx);
XX = permute(reshape(XX,nkt,nkx,nkt,[]),[1 3 2 4]);  % make 4th order tensor

% Calculate first two response moments
Ymu = sum(Resp);
YY = Resp'*Resp;
