function [Y,Stim,kTrue,kxOrth,ktOrth] = mkdata_1Dspatial_rank2(Nsamps,nkx,nkt,signse,noisetype)
% [Y,Stim,kTrue,dtbin,tmax,kxOrth,ktOrth] = mkdata_1DspatialRF_rank2(Nsamps,nkx,nkt,signse,noise)
% 
% Make a simulated dataset with 1D spatial stimulus and rank-2 stimulus filter.
%
% Inputs:
% -------
%     Nsamps - total number of stimulus samples
%        nkx - number of pixels in 1D spatial filter
%        nkt - number of time bins in stimulus filter
%      dtbin - length of a single time bin (in s)
%     signse - standard deviation of additive Gaussian noise
%  noisetype - select type of noise as 'white' or 'AR1'
% 
% Outputs:
% --------
%       Y [Nsamps x  1 ] - neural response vector
%    Stim [Nsamps x nkx] - spatio-temporal stimulus
%   kTrue [  nkt  x nkx] - spatio-temporal stimulus filter
%    tmax [    1  x  1 ] - length of temporal component (in s)
%  kxOrth [  nkx  x  2 ] - spatial components of filter
%  ktOrth [  nkt  x  2 ] - temporal components of filter

%% Step 1: generate example filters

%%%%%%%%% make temporal filters %%%%%%%%%%%%%%%%%%%%
 
tt = (nkt:-1:1)'/nkt; % time bins for temporal filter
kt1 = gampdf(tt,12,.05); kt1 = kt1/norm(kt1); % 1st temporal filter
kt2 = gampdf(tt,4,.1); kt2 = -kt2./norm(kt2); % 2nd temporal filter
kt = [kt1 kt2]; % concatenate temporal RFs into n x 2 matrix

dc = 0; % response offset (additive constant)

%%%%%%%%% make spatial filter %%%%%%%%%%%%%%%%%%%%

xx = linspace(-2,2,nkx)';
% Gabor RFs
kx1 = cos(2*pi*xx/2 + pi/5).*exp(-1/(2*0.35^2)*xx.^2);
kx2 = sin(2*pi*xx/2 + pi/5).*exp(-1/(2*0.35^2)*xx.^2);
kx1 = 1.2*kx1./norm(kx1); % rescale component
kx2 = kx2./norm(kx2); % rescale component
% concatenate
kx = [kx1 kx2];

%%%%%%%  plot rank-2 STRF example %%%%%%%%%%%%%%%%
kTrue = kt*kx'; % make full 2D filter

N = Nsamps; % sample size

switch noisetype
    case 'white'
        Sigma = eye(nkx); % independent (white noise) stimulus covariance
    case 'AR1'
        Sigma = toeplitz(exp(-(0:nkx-1)/(nkx/6))); % correlated (AR1) stim covariance
end
% % change marginal variances of diff pixels, if desired
% A = diag(2.^(.5/nkx).^(0:nkx-1));  
% Sigma = A*Sigma*A;

mu = zeros(nkx,1); % stimulus mean
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
