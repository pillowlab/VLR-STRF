function m = UpdateTemporalParams(m)
% function m = UpdateTemporalParams(m);
% 
% function to perform updates to hyperparameters and variational parameters
% related to the temporal receptive field

maxiter = m.opts.maxiter.tempStep;
optimArgs = {'display', 'none', 'maxiter', m.opts.maxiter.tempStep};
% opts = optimoptions('fmincon','Gradobj','on',optimArgs{:});
opts = optimset('Gradobj','on',optimArgs{:}, 'display', 'off');

[Q1,Q2] = CalculateMoments_TemporalParams(m);

m.ProjStats.Q1 = Q1; % store for use in model parameter update step

% build temporal Basis
hprstVec = getHyperParams(m.tempPrior); % get temporal parameters as a vector
[Bkt, ~, BasisInfo_Time] = m.tempPrior.BasisFun(hprstVec,m.tempPrior); % time basis

% build spatial basis
hprsxVec = getHyperParams(m.spatPrior); % get spatial hyperparameters as a vector
[Bkx,~,BasisInfo] = m.spatPrior.BasisFun(hprsxVec(2:end),m.spatPrior); % space basis

llb = m.tempPrior.LowerOptimBounds;
uub = m.tempPrior.UpperOptimBounds;

rnk     = m.strfRank;
mux     = m.spatPost.Mean;
mut     = m.tempPost.Mean;
Sigmat  = m.tempPost.Cov;

ff = zeros(maxiter,1);

for k = 1:maxiter
        
    nt = size(Bkt,2);
    St = Sigmat + mut*mut';
    St = permute(reshape(St,[nt rnk nt rnk]),[1 3 2 4]);
    Mt = reshape(mut,[nt,rnk]);
        
    lfunc = @(prs)  LossMstep(prs,m,Q1,Q2,St,Mt,BasisInfo_Time);
%     DerivCheck(lfunc,prs);

    opts.verbose = 0;
    [hprstVec,ff(k)] = minConf_TMP(lfunc,hprstVec,llb,uub,opts);

    m.tempPrior.otherInput = update_otherInput(m.tempPrior,hprstVec);

    [Bkt, ~, BasisInfo_Time] = m.tempPrior.BasisFun(hprstVec, m.tempPrior);
    [mut,Sigmat] = Estep_TemporalParams(m.SufficientStats.XY(:,BasisInfo.idx),m.SufficientStats.Xmu(:,BasisInfo.idx),rnk,mux,Bkt,Bkx,m.NoiseStd,m.Offset,Q1);
    
    % check convergence
    if k >1 && abs(ff(k) - ff(k-1))< m.opts.abstol
        break;
    end

end

% update model structure with updated parameters
hprsStruct = setHyperParams(hprstVec,m.tempPrior.name);
m.tempPrior.hprs = hprsStruct;

% ff = ff(1:k);

% update model structure
m.tempPost.Mean = mut;
m.tempPost.Cov  = Sigmat;

end

% ==================================================================
function [obj,dd,trms] = LossMstep(prs,m,Q1,Q2,St,Mt,BasisInfo)

rnk = m.strfRank;
sigma = m.NoiseStd;

Q2m = Q2*Mt';

[Bkt,dBkt,BasisInfo] = m.tempPrior.BasisFun(prs, m.tempPrior, BasisInfo);

idx = BasisInfo.idx;

t1 = 0;
t2 = 0;
for i = 1:rnk
    Qbb = Bkt'*Q1(idx,idx,i,i)*Bkt;
    t1 = t1 + sum(sum(Qbb.*St(:,:,i,i)'));
    for j = i+1:rnk
        Qbb = Bkt'*Q1(idx,idx,i,j)*Bkt;
        t2 = t2 + sum(sum(Qbb.*St(:,:,j,i)'));
    end
end

obj = 1/(2*sigma^2)*t1 + 1/sigma^2*t2 - 1/sigma^2*sum(sum(Bkt'.*(Q2m)')); % <likelihood>_q

trms = [1/(2*sigma^2)*t1,  1/sigma^2*t2,- 1/sigma^2*sum(sum(Bkt'.*(Q2m)'))];

tpr1 = permute(zeros(size(vec(prs))),[2 3 1]); % 1 x 1 x numhprs
tpr2 = permute(zeros(size(vec(prs))),[2 3 1]); % 1 x 1 x numhprs

for i = 1:rnk
    Qbd = 2*Q1(idx,idx,i,i)*Bkt*St(:,:,i,i);
    tpr1 = tpr1 + sum(sum(dBkt .* Qbd,1),2);

    for j = i+1:rnk
        
        Qbd = Q1(idx,idx,i,j)*Bkt*St(:,:,j,i) + Q1(idx,idx,i,j)'*Bkt*St(:,:,i,j);
        tpr2 = tpr2 + sum(sum(dBkt .* Qbd,1),2);

    end
end

dd =  1/(2*sigma^2) * tpr1 + 1/sigma^2 * tpr2 - 1/sigma^2 * sum(sum(dBkt .* Q2m,1),2);
dd = dd(:);

end

