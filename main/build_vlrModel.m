function m = build_vlrModel(Y,X,rnk,spatPrior,tempPrior,opts)
% function  m = initialise_vlrModel(Y,X,nkx,nkt,tempPrior,spatPrior);
%
% function to initialise model structure for variational low-rank strf
% inference
% 
% inputs:
% ----------
%       Y           --- [T x 1] observed responses
%       X           --- [T x D] stimulus matrix
%       rnk         --- receptive field rank
%       spatPrior   --- structure for spatial components prior   
%       tempPrior   --- structure for temporal components prior
%       opts        --- option structure overriding defautls (optional)
%
% outputs:
% ----------
%       m           --- model structure to be passed into optimiser function
%
% See also: build_vlrPrior.m, fit_vlrModel.m
%
% Duncker & Pillow, 2018-2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% set option values
opts = setOpionValues(opts);
m.opts = opts;

%% save prior function handle stuff
m.spatPrior = spatPrior;
m.tempPrior = tempPrior;

% receptive field rank
m.strfRank = rnk;
m.N = size(Y,1);
% number of temporal receptive field coefficients
nkt = max(m.tempPrior.dims);
%% process input and store suficient statistics for later use
if m.opts.ReducedMoments
    [XY,Xmu,XX,Ymu,YY] = CalculateMoments_reduced(Y,X,nkt);
else
    [XY,Xmu,XX,Ymu,YY] = CalculateMoments_full(Y,X,m.nkt);
end
% store sufficent stats
m.SufficientStats.XY    = XY;
m.SufficientStats.Xmu   = Xmu;
m.SufficientStats.XX    = XX;
m.SufficientStats.Ymu   = Ymu;
m.SufficientStats.YY    = YY;
m.SufficientStats.Y     = Y; 
m.SufficientStats.X     = X;

% Make sparse matrix for summing diagonals of an nkt x nkt matrix
m.Msum = mkDiagSummingMtx(nkt);

%% initialise model parameters
%[ss,bb] = initialise_ModelParams(Y);
m.NoiseStd  = std(Y);
m.Offset    = mean(Y);

%% initialise variational parameters
m = initialise_VariationalParams(m);

