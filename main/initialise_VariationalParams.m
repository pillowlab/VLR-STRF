function m = initialise_VariationalParams(m)
% m = initialise_VariationalParams(m)
% 
% Initializes variational parameters for low-rank RF model

% build temporal Basis
hprstVec = getHyperParams(m.tempPrior); % get temporal parameters as a vector
Bkt = m.tempPrior.BasisFun(hprstVec,m.tempPrior); % time basis

% build spatial basis
hprsxVec = getHyperParams(m.spatPrior); % get spatial hyperparameters as a vector

[Bkx,~,BasisInfo] = m.spatPrior.BasisFun(hprsxVec(2:end),m.spatPrior); % space basis

% extract dims
[nkt,nt] = size(Bkt);
[nkx,nx]= size(Bkx);
rnk = m.strfRank;

% moment based initialisations
[uu,~,vv] = svd(m.SufficientStats.XY);
mut = (Bkt'*Bkt+.1*eye(size(Bkt,2)))\(Bkt'*uu(1:size(Bkt,1),1:rnk));
mux = (Bkx'*Bkx+.1*eye(size(Bkx,2)))\(Bkx'*vv(1:size(Bkx,1),1:rnk)); 

% ------- project second order stimuli onto bases  -------------

% Project stimuli onto spatial basis
XXprjx = reshape(Bkx'*reshape(m.SufficientStats.XX(BasisInfo.idx,BasisInfo.idx,:),nkx,[]),nx,nkx,nkt); % left proj onto x basis
XXprjx = reshape(Bkx'*reshape(permute(XXprjx,[2 1,3]),nkx,[]),nx,nx,nkt); % right proj onto x basis

% Concatenate above and below-diagonal terms
zzx = [reshape(XXprjx,nx^2,nkt)'; ...
    flipud(reshape(permute(XXprjx(:,:,1:end-1),[2 1 3]),nx^2,nkt-1)')];

% Project onto spatial basis
XXprj = reshape(Bkt'*reshape(m.Msum'*zzx,nkt,[]),nt,nkt,nx^2); % left proj onto t basis
XXprj = reshape(Bkt'*reshape(permute(XXprj,[2 1 3]),nkt,[]),nt,nt,nx,nx); % right proj onto t basis
% Note: if above line is too expensive or gives out-of-memory errors, can
% rewrite with for loop to save memory, using one col of zzx at a time.

% Final permute and reshape into matrix
XXprj = reshape(permute(XXprj,[1 3 2 4]),nx*nt,nx*nt);

%
kk1 = kron(reshape(mux,[rnk nx])',eye(nt));
Sigmat = inv(eye(nt*rnk) + 1/m.NoiseStd^2*kk1'*XXprj*kk1);

m.tempPost.Mean = mut;
m.tempPost.Cov = Sigmat;
% ----------------------------------
% perform inference for spatial parameters first -- STA is probably worse
Q1 = CalculateMoments_SpatialParams(m);
[mux,Sigmax] = Estep_SpatialParams(m.SufficientStats.XY,m.SufficientStats.Xmu,rnk,mut,Bkt,Bkx,m.NoiseStd,m.Offset,Q1,BasisInfo);

m.spatPost.Mean = mux;
m.spatPost.Cov = Sigmax;

% ----------------------------------
% perform inference for temporal parameters, reassign Q1 to free up memory no longer needed

Q1 = CalculateMoments_TemporalParams(m);
[mut,Sigmat] = Estep_TemporalParams(m.SufficientStats.XY(:,BasisInfo.idx),m.SufficientStats.Xmu(:,BasisInfo.idx),rnk,mux,Bkt,Bkx,m.NoiseStd,m.Offset,Q1);

m.tempPost.Mean = mut;
m.tempPost.Cov = Sigmat;

