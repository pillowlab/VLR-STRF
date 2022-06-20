function [freeEnergy,trms] = variationalFreeEnergy(m)
% [freeEnergy,trms] = variationalFreeEnergy(m)
%
% Function to compute the value of the free energy 
%
rnk = m.strfRank;
Q1 = m.ProjStats.Q1;

% build temporal Basis
hprstVec = getHyperParams(m.tempPrior); % get temporal parameters as a vector
Bkt = m.tempPrior.BasisFun(hprstVec,m.tempPrior); % time basis
% build spatial basis
hprsxVec = getHyperParams(m.spatPrior); % get spatial hyperparameters as a vector
[Bkx,~,BasisInfo] = m.spatPrior.BasisFun(hprsxVec(2:end),m.spatPrior); % space basis

% get posteriors
mut = m.tempPost.Mean(:);
mux = m.spatPost.Mean(:);

Sigmat = m.tempPost.Cov;
Sigmax = m.spatPost.Cov;

nt = size(Bkt,2);
nx = size(Bkx,2);

Ait = eye(size(mut,1));
Aix = eye(size(mux,1));

St = Sigmat + mut*mut';
St = permute(reshape(St,[nt rnk nt rnk]),[1 3 2 4]);

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
trm1a = sum(sum(Mt.*((Bkt'*m.SufficientStats.XY(:,BasisInfo.idx)*Bkx)*Mx))); % dot product of filter with XY
trm1b = sum(sum(Mt.*((Bkt'*m.SufficientStats.Xmu(:,BasisInfo.idx)*Bkx)*Mx))); % dot product of filter with Xmu
trm1 = trm1a - trm1b*m.Offset;
trm2 = m.SufficientStats.YY - 2*m.SufficientStats.Ymu*m.Offset + m.Offset^2*m.N;

sigma = m.NoiseStd;
% Compute the terms 
t1 = -1/(2*sigma^2)*((t1 + 2*t2) - 2*trm1 + trm2) ...
    - m.N/2*log(2*pi*sigma^2); % <likelihood>_q

t21 = 0.5*logdet(Sigmat) + size(Sigmat,2)/2*log(2*pi*exp(1)); % entropy q(w_t)

t22 = 0.5*logdet(Sigmax) + size(Sigmax,2)/2*log(2*pi*exp(1)); % entropy q(w_x)

t31 = 0.5*logdet(Ait) - size(Ait,2)/2*log(2*pi) - 0.5*sum(sum(Ait.*(Sigmat + mut*mut')')); % <prior w_t>_q(w_t)

t32 = 0.5*logdet(Aix) - size(Aix,2)/2*log(2*pi) - 0.5*sum(sum(Aix.*(Sigmax + mux*mux')')); % <prior w_x>_q(w_x)

freeEnergy = t1 + t21 + t22 + t31 + t32;

trms = [t1 t21 t22 t31 t32];