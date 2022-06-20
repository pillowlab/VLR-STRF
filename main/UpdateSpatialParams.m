function m = UpdateSpatialParams(m)
% m = UpdateSpatialParams(m)
% 
% Performs interations of M-step and E-step updates for spatial RF parameters

maxiter = m.opts.maxiter.spatStep;
optimArgs = {'display', 'none', 'maxiter', m.opts.maxiter.spatStep};

% opts = optimoptions('fmincon','Gradobj','on',optimArgs{:});
opts = optimset('Gradobj','on',optimArgs{:}, 'display', 'off');

% build temporal Basis
hprstVec = getHyperParams(m.tempPrior); % get temporal parameters as a vector
[Bkt, ~, BasisInfo_Time] = m.tempPrior.BasisFun(hprstVec, m.tempPrior); % time basis

% build spatial basis
hprsxVec = getHyperParams(m.spatPrior); % get spatial hyperparameters as a vector

[Bkx,~,BasisInfo] = m.spatPrior.BasisFun(hprsxVec(2:end), m.spatPrior); % space basis


mut = m.tempPost.Mean(:);
mux = m.spatPost.Mean(:);
Sigmax = m.spatPost.Cov;
% extract STRF rank 
rnk = m.strfRank;

% biggest bottleneck (3s)
[Q1,Q2] = CalculateMoments_SpatialParams(m);

% set lower and upper bound for fmincon
llb = m.spatPrior.LowerOptimBounds(2:end);
uub = m.spatPrior.UpperOptimBounds(2:end);

ff = zeros(maxiter,1);

% don't optimise scale = 1 for spatial RF
hprsxVec = hprsxVec(2:end);

for k = 1:maxiter

    % get current number of (trimmed) fourier rf-coeffs
    nx = size(Bkx,2);  
    
    % get mean in matrix form
    Mx = reshape(mux,[rnk,nx])';
    
    % get covariance
    P = commutation(rnk,nx);
    Sx = P*(Sigmax + mux*mux')*P';
    Sx = permute(reshape(Sx,[nx rnk nx rnk]),[1 3 2 4]);
    
    % cost function for hyperparameter optimisation
    lfunc = @(prs) LossMstep(prs,m,Q1,Q2,Sx,Mx,BasisInfo);
     
%     DerivCheck(lfunc,hprsxVec)
    
    opts.verbose = 0;
    [hprsxVec,ff(k)] = minConf_TMP(lfunc,hprsxVec,llb,uub,opts);

    % update basis given new parameters
    [Bkx,~,BasisInfo]  = m.spatPrior.BasisFun(hprsxVec,m.spatPrior);

    [mux,Sigmax] = Estep_SpatialParams(m.SufficientStats.XY,m.SufficientStats.Xmu,rnk,mut,Bkt,Bkx,m.NoiseStd,m.Offset,Q1,BasisInfo);
    
    % check convergence
    if k >1 && abs(ff(k) - ff(k-1))< m.opts.abstol
        break;
    end
    
end
ff = ff(1:k);

% update model structure with updated parameters
hprsStruct = setHyperParams([1;hprsxVec],m.spatPrior.name);
m.spatPrior.hprs = hprsStruct;

% update model structure with updated posteriors
m.spatPost.Mean = mux;
m.spatPost.Cov  = Sigmax;
m.spatPrior.otherInput = update_otherInput(m.spatPrior, [1;hprsxVec]);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute loss function for M-step
function [obj,dd,trms] = LossMstep(prs,m,Q1,Q2,Sx,Mx,BasisInfo)

% build basis and gradients
[Bkx,dBkx,BasisInfo] = m.spatPrior.BasisFun(prs,m.spatPrior,BasisInfo);

idx = BasisInfo.idx;

rnk = m.strfRank;
sigma = m.NoiseStd;

Q2m = Mx*Q2(:,idx);

t1 = 0;
t2 = 0;
for i = 1:rnk
    Qbb = Bkx'*Q1(idx,idx,i,i)*Bkx;
    t1 = t1 + sum(sum(Qbb.*Sx(:,:,i,i)'));
    for j = i+1:rnk
        Qbb = Bkx'*Q1(idx,idx,j,i)*Bkx;
        t2 = t2 + sum(sum(Qbb.*Sx(:,:,i,j)'));
    end
end

obj = 1/(2*sigma^2)*t1 + 1/sigma^2*t2 - 1/sigma^2*sum(sum(Q2m.*Bkx')); % <likelihood>_q

trms = [1/(2*sigma^2)*t1,  1/sigma^2*t2,- 1/sigma^2*sum(sum(Q2m.*Bkx'))];

tpr1 = permute(zeros(size(vec(prs))),[2 3 1]); % 1 x 1 x numhprs
tpr2 = permute(zeros(size(vec(prs))),[2 3 1]); % 1 x 1 x numhprs

for i = 1:rnk
    
    Qbd = 2*Q1(idx,idx,i,i)*Bkx*Sx(:,:,i,i);
    tpr1 = tpr1 + sum(sum(dBkx .* Qbd,1),2);
    
    for j = i+1:rnk
        
        Qbd = Q1(idx,idx,j,i)*Bkx*Sx(:,:,i,j) + Q1(idx,idx,j,i)'*Bkx*Sx(:,:,i,j)';
        tpr2 = tpr2 + sum(sum(dBkx .* Qbd,1),2);
        
    end
end

dd =  1/(2*sigma^2) * tpr1 + 1/sigma^2 * tpr2 - 1/sigma^2 * sum(sum(permute(dBkx,[2 1 3]) .* Q2m,1),2);

% reshape into vector
dd = dd(:);

end

