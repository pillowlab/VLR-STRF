function [Bkx,dBkx,BasisInfo] = PriorBasis_ASD(hprs, Prior, BasisInfo, varagin);
% [Bkx,dBkx] = PriorBasis_TRD(hprs,tempPrior)
%
% Compute basis for TRD prior covariance matrix (and its derivative), such
% that samples from the prior can be computed via:
%
%       w_x = Bkx*eps,  where eps ~ N(0, I) is a standard normal.
% 
% Input
% -----
%       hprs - vector of hyperparameters  [scale, lens]
%  	    Prior - struct with basis information
%
% Output
% ------
%       Bkx [nkt, m] - basis
%   dBkx [nkt, m, 3] - derivative of basis w.r.t hyperparameters


% spaceFlag indicates if rho=1 fixed for spatial and hprs input shorter
dims = Prior.dims;

if nargin < 3
    BasisInfo = [];
end

% Build Basis Info if necessary
if isempty(BasisInfo)

    % TODO: pass this in as argument
    cutThresh = 1e-05;
    if any(dims ==1)
        BasisInfo.nxcirc = max(dims) + 3;
    else
       BasisInfo.nxcirc = dims + 3; 
    end
    BasisInfo.minl = 1; % one pixel is minimum lengthscale
    condthresh = 1e8; % threshold on condition number of covariance matrix

    maxw = floor((BasisInfo.nxcirc./(pi*BasisInfo.minl))*sqrt(.5*log(condthresh)));  % max freq to use

    if  any(dims == 1) % 1D case
        [U,ww] = realfftbasis(max(dims), maxw);

        % make covariance with minimum lengthscale and prune individual coefficients
        ccf =  makeASD(hprs,dims, ww, BasisInfo.nxcirc);
        ii = (ccf) > cutThresh; % indices of values to keep
        BasisInfo.wvec = ww(ii);% 1 D vector
        BasisInfo.U = U(ii,:);
        BasisInfo.idx = true(prod(dims),1);
        
    elseif ~any(dims == 1) % 2D case
        [U1,wvec1] = realfftbasis(dims(1), maxw(1));
        [U2,wvec2] = realfftbasis(dims(2), maxw(2));
        U = kron(U1,U2);
        [ww1,ww2] = ndgrid(wvec1,wvec2);
        ww = [ww1(:) ww2(:)];
        
        % make covariance with minimum lengthscale and prune individual coefficients
        ccf =  makeASD(hprs, dims, ww, BasisInfo.nxcirc);
        ii = (ccf) > cutThresh; % indices of values to keep

        BasisInfo.wvec = ww(ii,:); % 2 D vector
        BasisInfo.U = U(ii,:);
        BasisInfo.idx = true(prod(dims),1);
    end
end

[Bkx, dBkx] = makebasis(hprs,dims, BasisInfo);

end

%%%%%%%% helper function to compute basis and gradients %%%%%%%%
function kfdiag = makeASD(hprs, dims, ww, nxcirc);
% extract named hyperparameters from input
if length(hprs) == 2  && any(dims==1) % 1D case TIME
    lens = hprs(2); % lengthscale
    rho = hprs(1);
elseif length(hprs) == 2 && ~any(dims==1) % 2D case
    lens = hprs;
    rho =1;
elseif length(hprs) == 1  && any(dims==1) % 1D case SPACE
    lens = hprs(1); % lengthscale
    rho = 1;
else
    error('hyperparameter number and RF dimensionality mismatch')
end

if  any(dims == 1) % 1D case
    
    kfdiag = sqrt(2*pi)*rho*lens*exp(-(2*pi^2/nxcirc^2)*lens^2*ww.^2);
    
elseif ~any(dims == 1) % 2D case
    
    kfdiag = rho*(2*pi)*prod(lens)*exp(-((2*pi^2./nxcirc(:).^2).*lens(:).^2)'*ww')';
    
end


end

function [Bkx, dBkx] = makebasis(hprs, dims, BasisInfo);

Bfft = BasisInfo.U;
ww = BasisInfo.wvec;
nxcirc = BasisInfo.nxcirc;
kfdiag = makeASD(hprs, dims, BasisInfo.wvec, nxcirc);
% build fourier coefficients
S = sqrt(kfdiag);   % square rooted fourier coefficients
% Bkx = Bfft'*diag(S);  % low-rank Basis: Bb*Bb' = GP-prior covariance

Bkx = Bfft'.*S';

% Compute gradients if necessary
if nargout > 1
    
    
    if length(hprs) == 2  && any(dims==1) % 1D case TIME
        lens = hprs(2); % lengthscale
        rho = hprs(1);
    elseif length(hprs) == 2 && ~any(dims==1) % 2D case fixed rho
        lens = hprs;
        rho = 1;
        
    elseif length(hprs) == 1  && any(dims==1) % 1D case SPACE
        lens = hprs(1); % lengthscale
        rho = 1;
    else
        error('hyperparameter number and RF dimensionality mismatch')
    end

    % calculate gradients wrt hyperparameters
    ddr = 0.5*Bfft'.*S'/rho;
    
    % concatenate hyperparameter gradients
    if length(hprs) == 2  && any(dims==1) % 1D case TIME
        ddl = Bfft'.*(S.*(0.5*1/lens - 2*pi^2/nxcirc^2*lens*ww.^2))';
        dBkx = cat(3,ddr,ddl);
        
    elseif length(hprs) == 1 && any(dims==1) % 1D case rho fixed 
        dBkx = Bfft'.*(S.*(0.5*1/lens - 2*pi^2/nxcirc^2*lens*ww.^2))';

    elseif length(hprs) == 2  && ~any(dims==1) % 2D case SPACE
        dBkx1 = Bfft'.*(S.*(0.5*1/lens(1) - ww(:,1)*((2*pi^2./nxcirc(1).^2).*lens(1))))';
        dBkx2 = Bfft'.*(S.*(0.5*1/lens(2) - ww(:,2)*((2*pi^2./nxcirc(2).^2).*lens(2))))';
        dBkx = cat(3, dBkx1, dBkx2);
    else
        error('hyperparameter number and RF dimensionality mismatch')
    end
        
end

end
