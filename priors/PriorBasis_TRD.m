function [Bkt,dBkt, BasisInfo] = PriorBasis_TRD(hprs,tempPrior,varargin)
% [Bkt,dBkt] = PriorBasis_TRD(hprs,tempPrior)
%
% Compute basis for TRD prior covariance matrix (and its derivative), such
% that samples from the prior can be computed via:
%
%       w_t = Bkt*eps,  where eps ~ N(0, I) is a standard normal.
% 
% Input
% -----
%       hprs - vector of hyperparameters  [warp, scale, lens]
%  tempPrior - struct with basis information
%
% Output
% ------
%       Bkt [nkt, m] - temporal basis
%   dBkt [nkt, m, 3] - derivative of temporal basis w.r.t hyperparameters

% Extract inputs
dims = tempPrior.dims;
minlen = tempPrior.otherInput.minlen;
Tcirc = tempPrior.otherInput.Tcirc;
tmax = tempPrior.otherInput.tmax;
dt = tmax/max(dims);

BasisInfo.idx = true(max(dims),1);

% check that dimensionality is correct
assert(length(hprs) == 3) % correct number of hyperparameters
assert(any(dims == 1) || length(dims) == 1) % correct number of dimensions, 1D only

% rescale time
% tt = flipud((0:max(dims)-1)');
tt = (tmax:-dt:dt)';

ttwarp = tmax./log(1+exp(hprs(3))*tmax)*log(1 + exp(hprs(3))*tt);
% build covariance matrix

condthresh = 1e8; % threshold on condition number of covariance matrix

% set up Fourier frequencies and generate (warped) clear all Fourier basis
maxw = floor((Tcirc/(pi*minlen))*sqrt(.5*log(condthresh)));  % max freq to use

nw = maxw*2+1; % number of fourier frequencies
[Bfft,wvec,wcos,wsin] = realnufftbasis(ttwarp,Tcirc,nw); % make basis (and get Fourier freqs)

% build fourier coefficients
kfdiag = sqrt(2*pi)*hprs(1)*hprs(2)*exp(-(2*pi^2/Tcirc^2)*hprs(2)^2*wvec.^2);
S = sqrt(kfdiag);   % square rooted fourier coefficients
% Bkt = Bfft'*diag(S);  % low-rank Basis: Bb*Bb' = GP-prior covariance
Bkt = Bfft'.*S';

% Compute gradients if necessary
if nargout > 1
    
    % calculate gradients wrt hyperparameters
    ddr = 0.5*Bfft'.*S'/hprs(1);
    
    ddl = Bfft'.*(S.*(0.5*1/hprs(2) - 2*pi^2/Tcirc^2*hprs(2)*wvec.^2))';
    
    dtc = -tmax^2*exp(hprs(3))/(log(1 + exp(hprs(3))*tmax))^2 *(log(1 + exp(hprs(3))*tt)./(1 + exp(hprs(3))*tmax)) + ...
        tmax/log(1 + exp(hprs(3))*tmax) * exp(hprs(3)).*tt./(1 + exp(hprs(3))*tt);
    
    ddc  = ([-sin((2*pi/Tcirc)*wcos*ttwarp').*((2*pi/Tcirc)*wcos*dtc'); ...
        cos((2*pi/Tcirc)*wsin*ttwarp').*((2*pi/Tcirc)*wsin*dtc')]/sqrt(Tcirc/2))'.*S';
    
    % concatenate hyperparameter gradients
    dBkt = cat(3,ddr,ddl,ddc);
    
end


