function [khatALD, timings] = runALDestimation(y,x,nkt,spatialdims);

datastruct = formDataStruct(x, y, nkt, spatialdims);

numb_dims = length(datastruct.ndims);

% ml
% kml = datastruct.xx\datastruct.xy;  

%% Ridge regression for initialization

opts0.maxiter = 1000;  % max number of iterations
opts0.tol = 1e-6;  % stopping tolerance
lam0 = 10;  % Initial ratio of nsevar to prior var (ie, nsevar*alpha)
% ovsc: overall scale, nasevar: noise variance
t_start_ridge = tic;
[kRidge, ovsc ,nsevar]  =  runRidge(lam0, datastruct, opts0);

timings.ridge = toc(t_start_ridge);

%% 1. ALDs

% options: ALDs uses trust-region algorithm with analytic gradients and Hessians
% opts1 = optimset('display', 'iter', 'gradobj', 'on', 'hessian', 'on','tolfun',1e-8, 'TolX', 1e-8, 'TolCon', 1e-8, 'MaxIter', 1e3, 'MaxFunEval', 3*1e3);
opts1 = optimset('display', 'iter', 'gradobj', 'on','tolfun',1e-8, 'TolX', 1e-8, 'TolCon', 1e-8, 'MaxIter', 1e3, 'MaxFunEval', 3*1e3);

% Find good initial values
InitialValues = NgridInit_pixel(datastruct.ndims, nsevar, ovsc, kRidge); % make a coarse grid
prs_p = compGriddedLE(@gradLogEv_ALDs, InitialValues, datastruct); % evaluate evidence on the grid

t_start_ALDs = tic;
[khatALD.khatS, khatALD.evidS, khatALD.thetaS, khatALD.postcovS] = runALDs(prs_p, datastruct, opts1);
timings.ALDs = toc(t_start_ALDs);

%% 2. ALDf

% ALDf and ALDsf uses active-set algorithm with analytic gradients
opts2 = optimset('display', 'iter', 'gradobj', 'on', 'algorithm','active-set','tolfun',1e-8, 'TolX', 1e-8, 'TolCon', 1e-8, 'MaxIter', 1e3, 'MaxFunEval', 3*1e3);

% Initialize diagonal of M
InitialValues = NgridInit_freq_diag(datastruct.ndims, nsevar, khatALD.thetaS(end)); % make a coarse grid
prs_f = compGriddedLE(@gradPrior_ALDf_diag, InitialValues, datastruct); % evaluate evidence on the grid
% Run ALDf using diagonal M and zero mean to initialize M
[khatALD.khatFdiag, khatALD.evidFdiag, khatALD.thetaFdiag, khatALD.postcovFdiag] = runALDf_diag(prs_f, datastruct, opts2);

mu_init = zeros(numb_dims,1);
if numb_dims==1
    offDiagTrm = [];
else
    offDiagTrm = numb_dims*(numb_dims-1)/2;
end
prsALDf_init = [khatALD.thetaFdiag(1:end-1); 0.1*ones(offDiagTrm,1); mu_init; khatALD.thetaFdiag(end)];

t_start_ALDf = tic;
[khatALD.khatF, khatALD.evidF, khatALD.thetaF, khatALD.postcovF] = runALDf(prsALDf_init, datastruct, opts2);
timings.ALDf = toc(t_start_ALDf);
