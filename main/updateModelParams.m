function m = updateModelParams(m)
% function m = updateModelParams(m)
% 
% function to perform update constant offset and noise stdev
%

optimArgs = {'display', 'none', 'maxiter', m.opts.maxiter.Mstep};
opts    = optimoptions('fmincon','Algorithm','trust-region-reflective','Gradobj','on','Hessian','on',optimArgs{:});
% opts = optimset('Gradobj','on',optimArgs{:}, 'display', 'off');

% bounds for constrained opt
ll      = [-Inf;1e-06];
uu      = [Inf;Inf];

% inital param value
prs0    = [m.Offset;m.NoiseStd];

% build temporal Basis
hprstVec = getHyperParams(m.tempPrior); % get temporal parameters as a vector
Bkt = m.tempPrior.BasisFun(hprstVec,m.tempPrior); % time basis
% build spatial basis
hprsxVec = getHyperParams(m.spatPrior); % get spatial hyperparameters as a vector
[Bkx,~,BasisInfo] = m.spatPrior.BasisFun(hprsxVec(2:end),m.spatPrior); % space basis

% get dimensions
nt      = size(Bkt,2);
nx      = size(Bkx,2);
rnk     = m.strfRank;

% posterior means and covariance for temporal rf
Sigmat  = m.tempPost.Cov;
mut     = m.tempPost.Mean(:);
mux     = m.spatPost.Mean(:);

% get expected 2nd moment of temporal filter
St      = Sigmat + mut*mut';
St      = permute(reshape(St,[nt rnk nt rnk]),[1 3 2 4]);

% extract other stored statistic
Q1      = m.ProjStats.Q1;
YY      = m.SufficientStats.YY;
Ymu     = m.SufficientStats.Ymu;
XY      = m.SufficientStats.XY(:,BasisInfo.idx);
Xmu     = m.SufficientStats.Xmu(:,BasisInfo.idx);

t1 = 0;
t2 = 0;
for i = 1:rnk
    Qbb = Bkt'*Q1(:,:,i,i)*Bkt;
    t1 = t1 + sum(sum(Qbb.*St(:,:,i,i)'));
    for j = i+1:rnk
        Qbb = Bkt'*Q1(:,:,i,j)*Bkt;
        t2 = t2 + sum(sum(Qbb.*St(:,:,j,i)'));
    end
end

% Terms we need

Mx = reshape(mux,[rnk,nx])';  % spatial filters
Mt = reshape(mut,[nt,rnk]);  % temporal filters

trm1a = sum(sum(Mt.*((Bkt'*XY*Bkx)*Mx))); % dot product of filter with XY
trm1b = sum(sum(Mt.*((Bkt'*Xmu*Bkx)*Mx))); % dot product of filter with Xmu

% Loss function (negative log-likelihood for dc and sigma)
lfunc = @(prs) loss_ModelParams(prs,t1,t2,trm1a,trm1b,YY,Ymu,m.N);
% DerivCheck(lfunc,prs0)

% Optimize
prs = fmincon(lfunc,prs0,[],[],[],[],ll,uu,[],opts);
% prs = minFunc(lfunc,prs0,opts);

% update parameter
m.Offset = prs(1);
m.NoiseStd = prs(2);

end

function [neglogli,dneglogli,H] = loss_ModelParams(prs,t1,t2,trm1a,trm1b,YY,Ymu,N)
% [neglogli,dneglogli,H] = loss_ModelParams(prs,t1,t2,trm1a,trm1b,YY,Ymu,N)
% 
% Compute negative log-likelihood for dc and sigma_noise

dc = prs(1);
sigma = prs(2);

%% compute these terms 

trm1 = trm1a - trm1b*dc;
trm2 = YY - 2*Ymu*dc + dc^2*N;

%% Compute loss

% replacements
neglogli = 1/(2*sigma^2)*((t1 + 2*t2) - 2*trm1 + trm2) ...
    + N/2*log(sigma^2); % <likelihood>_q

% Compute gradient
if nargout > 1
    ddc =  1/(sigma^2)*(trm1b - Ymu + N*dc);
    dsig = - 1/(sigma^3)*((t1 + 2*t2) - 2*trm1 + trm2) + N/sigma;
    dneglogli = [ddc; dsig];        
end

% Compute Hessian
if nargout > 2
    Hb = N/sigma^2;
    Hsig =  3/(sigma^4)*((t1 + 2*t2) - 2*trm1 + trm2) - N/sigma^2;
    Hsigb = -2/sigma^3*(trm1b - Ymu + N*dc);
    H = [Hb Hsigb; Hsigb Hsig];
end

end