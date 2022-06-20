function [Y,Stim,kTrue,kxOrth,ktOrth, Sigma] = mkdata_2Dspatial_rank2(Nsamps,xdims,nkt,signse,noisetype)
% [Y,Stim,kTrue,kx,kt] = mkdata_2Dspatial_rank2(Nsamps,xdims,nkt,signse,noisetype)
% 
% Make a simulated dataset with 2D spatial stimulus and rank-2 stimulus filter.
%
% Inputs:
% -------
%         Nsamps - total number of stimulus samples
%  xdims [nx ny] - dimensions of 2D spatial filter
%            nkt - number of time bins in stimulus filter
%         signse - standard deviation of additive Gaussian noise
%      noisetype - select type of noise as 'white' or 'AR1'
% 
% Outputs:
% --------
%       Y [Nsamps x  1 ] - neural response vector
%    Stim [Nsamps x nkx] - spatio-temporal stimulus
%   kTrue [  nkt  x nkx] - spatio-temporal stimulus filter (as a matrix)
%    tmax [    1  x  1 ] - length of temporal component (in s)
%  kxOrth [  nkx  x  2 ] - spatial components of filter
%  ktOrth [  nkt  x  2 ] - temporal components of filter

%% Step 1: generate example filters

%%%%%%%%% make temporal filters %%%%%%%%%%%%%%%%%%%%
tt = (nkt:-1:1)'/nkt; % time bins for temporal filter
kt1 = gampdf(tt,12,.05); kt1 = kt1/norm(kt1); % 1st temporal filter
kt2 = gampdf(tt,4,.1); kt2 = -kt2./norm(kt2); % 2nd temporal filter
kt = [kt1 kt2]; % concatenate temporal RFs into n x 2 matrix

dc = -1; % response offset (additive constant)

%%%%%%%%% make spatial filter %%%%%%%%%%%%%%%%%%%%

% make grid for spatial pixels
[xx1,xx2] = ndgrid(linspace(-2,2,xdims(1)),linspace(-2,2,xdims(2))');
xx1 = xx1(:);  xx2 = xx2(:); % make column vectors

% make two Gabor filters
kx1 = MakeGaborFilter(xx1,xx2,1,1.5,1.5,2*pi/9,0,xdims(1),xdims(2));
kx2 = MakeGaborFilter(xx1,xx2,1,1.5,0.7,1.5*pi/9,1,xdims(1),xdims(2));
kx = [kx1(:) kx2(:)];  % concatenate spatial RFs into nkx x 2 matrix

% make rank-2 STRF example
kTrue = kt*kx';

%%
N = Nsamps; % sample size

nkx = prod(xdims);
switch noisetype
    case 'white'
        Sigma = eye(nkx); % independent (white noise) stimulus covariance
    case 'AR1'
        Sigma = toeplitz(exp(-(0:nkx-1)/(nkx/6))); % correlated (AR1) stim covariance
end
% % change marginal variances of diff pixels, if desired
% A = diag(2.^(.5/nkx).^(0:nkx-1));  
% Sigma = A*Sigma*A;

mu = zeros(1,nkx); % stimulus mean
Stim = mvnrnd(mu,Sigma,N); % generate stimulus
filterResp = sameconv(Stim,kTrue) + dc;
Y =  filterResp + signse*randn(N,1);

% Report SNR
SNR = var(filterResp)/signse.^2;
fprintf('SNR = %.3f\n', SNR);

% Find orthogonal basis for filters
[uu,ss,vv] = svd(kTrue);
ktOrth = uu(:,1:2)*sqrt(ss(1:2,1:2));
kxOrth = vv(:,1:2)*sqrt(ss(1:2,1:2));
