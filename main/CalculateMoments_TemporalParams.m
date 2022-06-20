function [Q1,Q2] = CalculateMoments_TemporalParams(m)
% [Q1,Q2] = CalculateMoments_TemporalParams(m)
%
% Computes moments needed for updating temporal parameters

hprsVec = getHyperParams(m.spatPrior);
[Bkx,~,BasisInfo] = m.spatPrior.BasisFun(hprsVec(2:end), m.spatPrior);

mux = m.spatPost.Mean(:);
Sigmax = m.spatPost.Cov;

rnk = m.strfRank;

[nkt,nkx] = size(m.SufficientStats.XY(:,BasisInfo.idx));
nx = size(Bkx,2);

% =====================================================================
% Compute Q1: 4th order tensor w/ blocks of 2nd-order stim moments 

% make block-diagonal matrix from temporal basis
Bkxblock = repdiag(Bkx,rnk);  

% Project covariance matrix up to pixel space using temporal basis
P = commutation(rnk,nx);
Sx = Bkxblock*(P*(Sigmax + mux*mux')*P')*Bkxblock'; 
Sx = reshape(permute(reshape(Sx,[nkx rnk nkx rnk]),[1 3 2 4]),nkx^2,rnk^2); % reshape

if m.opts.ReducedMoments
    % Compute Q1
    dd1 = reshape(m.SufficientStats.XX(BasisInfo.idx,BasisInfo.idx,:),nkx^2,nkt)'*Sx;  % below-diagonal terms
    dd2 = reshape(permute(m.SufficientStats.XX(BasisInfo.idx,BasisInfo.idx,1:end-1),[2 1 3]),nkx^2,nkt-1)'*Sx; % above diagonal terms
    dd = [dd1; flipud(dd2)]; % diagonal values for each term
    Q1 = reshape(m.Msum'*dd,nkt,nkt,rnk,rnk); % put them into diagonal matrices
    
else
    % Compute Q1
    XXr = reshape(m.SufficientStats.XX(:,BasisInfo.idx),nkt^2, nkx^2);
    Q1 = reshape(XXr*Sx,nkt,nkt,rnk,rnk);
    
end

% =====================================================================
% Compute Q2: weighted spike-triggered matrix
Q2 = (m.SufficientStats.XY(:,BasisInfo.idx)-m.Offset*m.SufficientStats.Xmu(:,BasisInfo.idx))*Bkx*reshape(mux,[rnk,nx])';



