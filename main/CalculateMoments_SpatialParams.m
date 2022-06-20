function [Q1,Q2] = CalculateMoments_SpatialParams(m)

% make basis given current parameter settings
Bkt     = m.tempPrior.BasisFun(getHyperParams(m.tempPrior),m.tempPrior);
rnk     = m.strfRank;
% Get sizes
[nkt,nt] = size(Bkt);
nkx = size(m.SufficientStats.XY,2);

mut     = m.tempPost.Mean(:);
Sigmat  = m.tempPost.Cov;

% =====================================================================
% Compute Q1: 4th order tensor w/ blocks of 2nd-order stim moments 

% make block-diagonal matrix from temporal basis
Bktblock = repdiag(Bkt,rnk);  

% Project covariance matrix up to pixel space using temporal basis
St = Bktblock*(Sigmat + mut*mut')*Bktblock'; 

if m.opts.ReducedMoments
    dd = m.Msum*reshape(permute(reshape(St,[nkt rnk nkt rnk]),[1 3 2 4]),nkt^2,rnk^2);
    
    % Now weight the moments in XXred by diag values dd
    Q1 = reshape(m.SufficientStats.XX,nkx^2,nkt)*dd(1:nkt,:) + ...  % below diagonal terms
        reshape(permute(m.SufficientStats.XX(:,:,1:end-1),[2,1,3]),nkx^2,nkt-1)*flipud(dd(nkt+1:end,:)); % above diagonal terms
    
    % Reshape it
    Q1 = reshape(Q1,nkx,nkx,rnk,rnk);
    
    % =====================================================================
    % Compute Q2: projected 1st moments

    Q2 = reshape(mut,[nt,rnk])'*Bkt'*(m.SufficientStats.XY-m.Offset*m.SufficientStats.Xmu);
else
    St = reshape(permute(reshape(St,[nkt rnk nkt rnk]),[1 3 2 4]),nkt^2,rnk^2); % reshape to tensor
    
    % Compute each block of Q1
    XXr = reshape(m.SufficientStats.XX,nkt^2, nkx^2);
    Q1 = reshape(XXr'*St,nkx,nkx,rnk,rnk);
    
    % =====================================================================
    % Compute Q2: projected 1st moments
    
    Q2 = reshape(mut,[nt,rnk])'*Bkt'*(m.SufficientStats.XY-m.Offset*m.SufficientStats.Xmu);
end

