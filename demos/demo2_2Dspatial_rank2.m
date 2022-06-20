% demo script for new VLR code base
clear;

% set random seed 
rng(2)

% set paths
codeDir1 = '~/Documents/git/VLR-STRF'; % YOUR CODE DIR HERE
codeDir2 = '~/Dropbox/Docs/git/VLR-STRF';  % YOUR CODE DIR HERE
try cd(codeDir1);
catch; cd(codeDir2);
end
set_paths;

%% Generate simulated dataset and compute STA

% Set simulation params
Nsamps = 1e04; % number of time samples to in stimulus
xdims = [20 20]; % spatial dimensions of stimulus
nkx = prod(xdims);  % total number of spatial RF coeffs
nkt = 25;  % length of temporal filter (in bins)
dtbin = 0.02; % lenth of a single time bin
tmax = nkt*dtbin; % length of temporal RF
signse = 2; % standard deviation of additive Gaussian noise
tt = (-tmax(1):dtbin:-dtbin)'; % time bins for filter

% Make simulated dataset
[Y,X,kTrue,kx,kt] = mkdata_2Dspatial_rank2(Nsamps,xdims,nkt,signse,'white');

% compute spike-triggered average (STA)
kSTA = simpleRevcorr(X,Y,nkt);

%% Make plots of true filters and STA
fprintf('\nPlotting STA estimate:\n');

subplot(321);
imagesc(1:nkx,[-tmax, 0],kTrue);colormap hot;
title('true STRF'); %xlabel('space (pixels)');
ylabel('time before spike (s)');
subplot(343);
imagesc(1:xdims(1),1:xdims(2), reshape(kx(:,1),xdims)); axis image;
xlabel('space dim 1'); ylabel('space dim 2'); title('x component 1');
subplot(344);
imagesc(1:xdims(1),1:xdims(2), reshape(kx(:,2),xdims)); axis image;
xlabel('space dim 1'); ylabel('space dim 2'); title('x component 2');

% SVD of STA
[tsta,~,xsta] = svd(kSTA);
xsta = xsta(:,1:2)*(xsta(:,1:2)\kx); % spatial components
tsta = tsta(:,1:2)*(tsta(:,1:2)\kt); % temporal components
subplot(323);
imagesc(1:nkx,[-tmax, 0],kSTA);colormap hot;
title('STA');
ylabel('time before spike (s)'); %xlabel('space (pixels)');
subplot(347);
imagesc(1:xdims(1),1:xdims(2), reshape(xsta(:,1),xdims)); axis image;
xlabel('space dim 1'); ylabel('space dim 2');
subplot(348);
imagesc(1:xdims(1),1:xdims(2), reshape(xsta(:,2),xdims)); axis image;
xlabel('space dim 1'); ylabel('space dim 2');

% plot components
subplot(325);
plot(1:nkx,kx,1:nkx,xsta,'--','linewidth',2); hold off;
axis tight;
title('spatial RF'); xlabel('space idx')
subplot(326);
h = plot(tt,kt,tt,tsta,'--','linewidth',2); hold off;
title('temporal RF');ylabel('time before spike (s)'); 
title('temporal components');
xlabel('time before spike (s)');
legend(h([1 3]),'true','est')
drawnow;

% Compute R^2 of STA
STAmse = sum((kTrue(:)/norm(kTrue(:)-kSTA(:)./norm(kSTA(:)))).^2);
STArsq = 1 - STAmse;
fprintf('STA R^2 = %1.3f \n\n',STArsq);


%% set up model fitting

% build prior for spatial RF
spatPrior = build_vlrPrior('ALD',xdims);
% spatPrior = build_vlrPrior('ASD',xdims);

% build prior for temporal RF
minlen_t = dtbin*2;   % minimum temporal lengthscale in normalised units
tempPrior = build_vlrPrior('TRD',nkt,minlen_t,tmax);

% update initial hyperparameters from STA
[tempPrior, spatPrior] = initialiseHprs_vlrPriors(kSTA,tempPrior,spatPrior);

%% initialise model structure

rnk = 2;            % receptive field rank
opts = [];          % use default options

% build model structure 
m = build_vlrModel(Y,X,rnk,spatPrior,tempPrior,opts); 

%% Fit low-rank STRF using variational EM

% Set number of iterations per step of coordinate ascent
m.opts.maxiter.spatStep = 10;
m.opts.maxiter.tempStep = 10;
m.opts.maxiter.EM = 20;  % total number of EM iterations

% Run variational EM
fprintf('\nRunning variational EM...\n-------------------------\n\n');
m = fit_vlrModel(m);

%% Extract MAP filter estimate and hyper-parameter estimate

xhprs_hat = m.spatPrior.hprs;  % extract fitted spatial hyperparams
thprs_hat = m.tempPrior.hprs;  % extract fitted temporal hyperparams

% get maximum a posteriori estimate in first output argument
[mutHat,muxHat]  = getSTRF_vlrModel(m);

kMAP = mutHat*muxHat';
mut = mutHat*(mutHat\kt);  % representation of true components in temporal basis
mux = muxHat*(muxHat\kx);  % representation of true components in spatial basis

% make plot
subplot(323);
imagesc(1:nkx,[-tmax, 0],kMAP);colormap hot;
title('MAP');
ylabel('time before spike (s)'); %xlabel('space (pixels)');
subplot(347);
imagesc(1:xdims(1),1:xdims(2), reshape(mux(:,1),xdims)); axis image;
xlabel('space dim 1'); ylabel('space dim 2');
subplot(348);
imagesc(1:xdims(1),1:xdims(2), reshape(mux(:,2),xdims)); axis image;
xlabel('space dim 1'); ylabel('space dim 2');

% plot components
subplot(325);
plot(1:nkx,kx,1:nkx,mux,'--','linewidth',2); hold off;
axis tight;
title('spatial RF'); xlabel('space idx')
subplot(326);
h = plot(tt,kt,tt,mut,'--','linewidth',2); hold off;
title('temporal RF');ylabel('time before spike (s)'); 
title('temporal components');
xlabel('time before spike (s)');
legend(h([1 3]),'true','estim','location','northwest');

%% report R-squared between true and estimated STRF
mse = sum((kTrue(:)-kMAP(:)).^2);
Rsq = 1 - mse./sum(kTrue(:).^2);
fprintf('R^2 = %1.3f \n',Rsq);

