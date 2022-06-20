function [Bkx,dBkx,BasisInfo] = PriorBasis_ALD(hprs,spatPrior, BasisInfo, cutFlag)
% [Bkx,BasisInfo,dBkx] = PriorBasis_ALD(hprs,dims,BasisInfo,cutFlag)
%
% Compute basis for ALD prior covariance matrix (and its derivative), such
% that samples from the prior can be computed via:
%
%       w = Bkx*eps,  where eps ~ N(0, I) is a standard normal.
%
% Inputs: 
% -------
%       hprs - hyperparameters (vector length 4 or 10 for 1D or 2D)--
%                  does not include scaling since fixed to 1
%       spartPrior - spatial prior structure
%    cutFlag - boolean for cutting small-prior-variance coeffs (1=cut)
%    basisFlag - boolean saying whether to use basis in prior struct or rebuild   
%
% Outputs:
% --------
%       Bkx - prior basis weighted by sqrt of singular values
% BasisInfo - struct with basis information
%      dBkx - derivative of basis

dims = spatPrior.dims;
% extract hyperparameters from input vector
if length(hprs) == 4  && any(dims==1)% 1D case
    pkxs = hprs(1); % inverse lengthscale
    vkxs = hprs(2); % mean
    mkxf = hprs(3); % inverse lengthscale
    vkxf = hprs(4); % mean
elseif length(hprs) == 10 && ~any(dims==1) % 2D case
    pkxs = hprs(1:3); % inverse lengthscale
    vkxs = hprs(4:5); % mean
    mkxf = hprs(6:8); % inverse lengthscale
    vkxf = hprs(9:10); % mean
else
    error('hyperparameter number and RF dimensionality mismatch')
end

% Set cutFlag if necessary
if nargin <= 3 || isempty(cutFlag)
    cutFlag = 1;
end

if nargin < 3
    BasisInfo = [];
end

% Build Basis Info if necessary
if isempty(BasisInfo)

    % TODO: pass this in as argument
    cutThresh = 1e-05;

    if  any(dims == 1) % 1D case
        [U,ww] = realfftbasis(max(dims));
        xx = (1:max(dims))';
        % make covariance with minimum lengthscale and prune individual coefficients
        [ccf,ccs] =  MakeALDsf(1,pkxs,vkxs,1,mkxf,vkxf,xx,ww);
        ii = (ccf) > cutThresh; % indices of values to keep
        if cutFlag == 1
            jj = sqrt(ccs) > cutThresh; % indices of values to keep
        else
            %fprintf(' %%%%%%%% Didnt CUT %%%%%%%%%%%%\n\n');
            jj = true(max(dims),1);
        end
        BasisInfo.wvec = ww(ii);% 1 D vector
        BasisInfo.xvec = xx(jj);
        BasisInfo.U = U(ii,jj);
        BasisInfo.idx = jj;
    elseif ~any(dims == 1) % 2D case
        [U1,wvec1] = realfftbasis(dims(1));
        [U2,wvec2] = realfftbasis(dims(2));
        U = kron(U1,U2);
        [ww1,ww2] = ndgrid(wvec1,wvec2);
        ww = [ww1(:) ww2(:)];
        [xx1,xx2] = ndgrid(1:dims(1),1:dims(2));
        xx = [xx1(:) xx2(:)];
        
        % make covariance with minimum lengthscale and prune individual coefficients
        [ccf,ccs] =  MakeALDsf(1,pkxs,vkxs,1,mkxf,vkxf,xx,ww);
        ii = (ccf) > cutThresh; % indices of values to keep
        if cutFlag == 1
            jj = sqrt(ccs) > cutThresh; % indices of values to keep
        else
            jj = true(prod(dims),1);
        end
        BasisInfo.wvec = ww(ii,:); % 2 D vector
        BasisInfo.xvec = xx(jj,:);
        BasisInfo.U = U(ii,jj);
        BasisInfo.idx = jj;
    end
end

% Compute basis itself
if nargout < 3
    Bkx = makebasis(pkxs,vkxs,mkxf,vkxf,BasisInfo);
else
    [Bkx,dBkx] = makebasis(pkxs,vkxs,mkxf,vkxf,BasisInfo);
end

end


% ================================================================
%%%%%%%% helper function to compute basis and gradients %%%%%%%%
function [Bb,dBb] = makebasis(ls,vs,mf,vf,BasisStruct)
% make ALD Basis
rs = 1;
rf = 1;
U = BasisStruct.U;
ww = BasisStruct.wvec;
xx = BasisStruct.xvec;

[ccf,ccs,diff,diffs,M,P] = MakeALDsf(rs,ls,vs,rf,mf,vf,xx,ww);

Sxs = sqrt(ccs);   % square rooted fourier coefficients 
Sxf = sqrt(ccf);   % square rooted fourier coefficients 

% Bb = diag(Sxs)*U'*diag(Sxf);  % low-rank Basis: Bb*Bb' = GP-prior covariance
Bb = (Sxs .* U').*Sxf';
if nargout > 1
    dMww  = bsxfun(@times,-0.5*diff,Sxf).*sign(ww*M);
    
    if length(ls) == 1 && length(vs) == 1 % 1D case
        
        ddvf = (Sxs.*U').*(bsxfun(@times,Sxf,(0.5*diff)))';
        
        ddmf = (Sxs.*U').*(dMww.*ww)';

        ddvs = 0.5*P*((diffs.*Sxs).*U').*(Sxf');

        ddps = (-0.5*(Sxs.*diffs.^2*ls).*U').*(Sxf');
        
        % assemble gradients
        dBb = cat(3,ddps,ddvs,ddmf,ddvf);
        
    else % 2D case
        
        ddvf1 = (Sxs.*U').*(Sxf.*(0.5*diff(:,1)))';
        ddvf2 = (Sxs.*U').*(Sxf.*(0.5*diff(:,2)))';
        ddmf1 = (Sxs.*U').*(dMww(:,1).*ww(:,1))';
        ddmf2 = (Sxs.*U').*(sum(dMww.*fliplr(ww),2))';
        ddmf3 = (Sxs.*U').*(dMww(:,2).*ww(:,2))';
        
        ddv = bsxfun(@times,Sxs',0.5*P*diffs')';
        ddvs1 = (ddv(:,1).*U').*(Sxf');
        ddvs2 = (ddv(:,2).*U').*(Sxf');
        ddps1 = 0.5*(Sxs.*(-diffs(:,1).^2*ls(1) + diffs(:,1).*diffs(:,2)*ls(2)*ls(3)).*U').*(Sxf');
        ddps2 = 0.5*(Sxs.*(diffs(:,1).*diffs(:,2)*ls(3)*ls(1)).*U').*(Sxf');
        ddps3 = 0.5*(Sxs.*(-diffs(:,2).^2*ls(3) + diffs(:,1).*diffs(:,2)*ls(1)*ls(2)).*U').*(Sxf');
        
        % assemble gradients
        dBb = cat(3,ddps1,ddps2,ddps3,ddvs1,ddvs2,ddmf1,ddmf2,ddmf3,ddvf1,ddvf2);
        
    end
end

end

% ================================================================
function [ccf,ccs,varargout] = MakeALDsf(rs,ls,vs,rf,mf,vf,xx,ww)
% function to construct diagonal matrices of ALDsf

[ccs,difx,P] = spaceALD(rs,ls,vs,xx);
[ccf,diff,M] = freqALD(rf,mf,vf,ww);

if nargout >3
    varargout{1} = diff;
    varargout{2} = difx;
    varargout{3} = M;
    varargout{4} = P;
end

end

% ================================================================
function [cc,diff,M] = freqALD(r,l,v,ww)
if size(v,2) > size(v,1)
    v = v'; % make column vector
end
    
if length(l) == 1 && length(v) == 1 % 1D case
    M = l;
elseif length(l) == 3 && length(v) == 2 % 2D case
    M = [l(1) l(2); l(2) l(3)];
end

diff = bsxfun(@minus,abs(M*ww'),v)';
cc = r.*exp(-0.5*sum(diff.^2,2));
end

function [cc,diff,P] = spaceALD(r,l,v,xx)
if size(v,2) > size(v,1)
    v = v'; % make column vector
end

if length(l) == 1 && length(v) == 1 % 1D case
    P =l^2;
elseif length(l) == 3 && length(v) == 2 % 2D case
    P = [l(1)^2, -l(1)*l(2)*l(3); -l(1)*l(2)*l(3), l(3)^2];
end

diff = bsxfun(@minus,xx,v');
cc = r.*exp(-0.5*sum(diff'.*((P*diff')),1)');
end


