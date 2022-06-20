% demo script for new VLR code base
clear;

% set random seed 
rng(2)

% set paths
codeDir1 = '~/Documents/git/VLR-STRF';  % YOUR CODE DIR HERE
codeDir2 = '~/Dropbox/Docs/git/VLR-STRF';  % YOUR CODE DIR HERE
try cd(codeDir1);
catch; cd(codeDir2);
end
set_paths;

%% Generate simulated dataset 

Nsamps = 1e04; % number of stimulus time samples
nkx = 250; % number of spatial pixels in RF
nkt = 25; % number of time bins in RF
dtbin = 0.2; % length of a single time bin (s)
tmax = nkt*dtbin; % total length of temporal RF (s)
signse = 10; % standard deviation of additive Gaussian noise
tt = (-tmax(1):dtbin:-dtbin)'; % time bins for filter

% Generate dataset
[Y,X,kTrue,kx,kt] = mkdata_1Dspatial_rank2(Nsamps,nkx,nkt,signse,'AR1');

% compute spike-triggered average (STA)
kSTA = simpleRevcorr(X,Y,nkt);
kSTA = kSTA/var(X(:))/Nsamps; % scaled so norm is upper bound on true norm

% Compute SVD of STA
[tsta,~,xsta] = svd(kSTA);
xsta = xsta(:,1:2)*(xsta(:,1:2)\kx); % reconstruction of spatial components
tsta = tsta(:,1:2)*(tsta(:,1:2)\kt); % reconstruction of temporal components

% Plot true filter and STA estimate
clf; 
figure;
subplot(231); 
imagesc(1:nkx,[-tmax, 0],kTrue);
colormap hot;title('true 2D STRF');
xlabel('space (pixels)'); ylabel('time before spike (s)');
subplot(232);
imagesc(1:nkx,[-tmax, 0],kSTA); title('STA'); 
subplot(234);
plot(1:nkx,kx(:,1),1:nkx,xsta(:,1),'--','linewidth',2); 
title('1st spatial comp'); xlabel('space idx')
subplot(235);
plot(1:nkx,kx(:,2),1:nkx,xsta(:,2),'--','linewidth',2); 
title('2nd spatial comp'); xlabel('space idx')
subplot(236);
plot(tt,kt,tt,tsta,'--','linewidth',2);
title('temporal components'); xlabel('time before spike (s)');
drawnow;

% Compute R^2 of STA estimate
STAmse = sum((kTrue(:)/norm(kTrue(:))-kSTA(:)./norm(kSTA(:))).^2);
STArsq = 1 - STAmse;
fprintf('STA R^2 = %1.3f \n\n',STArsq);

%% set up model priors

% build prior for spatial RF
RFdims = [nkx 1]; 
spatPrior = build_vlrPrior('ASD', RFdims);

% build prior for temporal RF
minlen_t = 5*dtbin;  % minimum temporal lengthscale (in s)
tempPrior = build_vlrPrior('TRD',nkt,minlen_t,tmax);

% update initial hyperparameters from STA
[tempPrior, spatPrior] = initialiseHprs_vlrPriors(kSTA,tempPrior,spatPrior);

%% initialise model structure

rnk = 2;   % receptive field rank
opts = []; % use default options

% build model structure 
m = build_vlrModel(Y,X,rnk,spatPrior,tempPrior,opts); 

%% Fit low-rank STRF using variational EM

% Set number of iterations per step of coordinate ascent
m.opts.maxiter.spatStep = 15;
m.opts.maxiter.tempStep = 15;
m.opts.maxiter.EM = 25;  % total number of EM iterations

% Run variational EM
fprintf('\nRunning variational EM...\n-------------------------\n\n');
tic;
m = fit_vlrModel(m);
toc;

%% Extract MAP filter estimate and hyper-parameter estimate

xhprs_hat = m.spatPrior.hprs;  % extract fitted hyperparams
thprs_hat = m.tempPrior.hprs;  % extract fitted hyperparams

% get maximum a posteriori estimate in first output argument
[mutHat,muxHat]  = getSTRF_vlrModel(m); % extract spatial and temporal filters from model
kMAP = mutHat*muxHat'; % full STRF reconstruction

 %% Plot fitted filters

mut = mutHat*(mutHat\kt);  % representation of true components in temporal basis
mux = muxHat*(muxHat\kx);  % representation of true components in spatial basis
tt = (-tmax(1):dtbin:-dtbin)'; % time axis

subplot(233);
imagesc(kMAP); title('VLR MAP')
subplot(234);
plot(1:nkx,kx(:,1),1:nkx,mux(:,1),'--','linewidth',2); 
title('1st spatial comp (MAP)'); xlabel('space idx')
subplot(235);
plot(1:nkx,kx(:,2),1:nkx,mux(:,2),'--','linewidth',2); 
title('2nd spatial comp (MAP)'); xlabel('space idx')
subplot(236);
h = plot(tt,kt,tt,mut,'--','linewidth',2);
title('temporal components');
xlabel('time before spike (s)');
legend(h([1 3]),'true','MAP','location','northwest');

%% report R-squared between true and estimated STRF
mse = sum((kTrue(:)-kMAP(:)).^2);
Rsq = 1 - mse./sum(kTrue(:).^2);
fprintf('R^2 = %1.3f \n',Rsq);
