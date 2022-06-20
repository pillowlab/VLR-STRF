function [tempPrior, spatPrior] = initialiseHprs_vlrPriors(k0,tempPrior,spatPrior, varargin)
% [tempPrior, spatPrior] = initialiseHprs_vlrPriors(k0,tempPrior,spatPrior)
% 
% Initializes model hyperparameters using spike triggered average
%
% INPUT:
%  k0 [nt x nx] - initial filter estimate (as a matrix), eg STA
%     spatPrior - structure specifying spatial prior
%     tempPrior - structure specifying temporal prior
%
% OUTPUT
%     spatPrior - structure specifying spatial prior with initialised
%                 hyperparamers
%     tempPrior - structure specifying temporal prior with initialised
%                 hyperparamers
%
% See also: build_vlrModel.m, fit_vlrModel.m
%
% Duncker & Pillow, 2018-2020
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initialize spatial parameters
if nargin > 3
    verbose = varargin{1};
else
    verbose =1;
end
if verbose
    fprintf('Initializing hyper-parameters\n-----------------------------\n');
end

switch spatPrior.name
    case 'ASD'
                
        InitialHprs_Space = initialiseHprs_ASD(k0,spatPrior.dims);

    case 'ALD'

        InitialHprs_Space = initialiseHprs_ALD(k0,spatPrior.dims);
        
    case 'RR'
        InitialHprs_Space = initialiseHprs_RR();

    case 'TRD'
        error('TRD can only be used as a temporal receptive field prior')
end

%% initialise temporal parameters
switch tempPrior.name
    case 'ASD'
        InitialHprs_Time = initialiseHprs_ASD(k0',tempPrior.dims);
        
    case 'TRD'

        InitialHprs_Time =  initialiseHprs_TRD(k0/5,tempPrior.otherInput.tmax);

    case 'RR'
        InitialHprs_Time = initialiseHprs_RR();
    
    case 'ALD'
        error('ALD can only be used as a spatial receptive field prior')
    
end

spatPrior.hprs = InitialHprs_Space;
tempPrior.hprs = InitialHprs_Time;

hprsxVec = getHyperParams(spatPrior); % get spatial hyperparameters as a vector
hprstVec = getHyperParams(tempPrior); % get spatial hyperparameters as a vector

% update other inputs that may depend on initial parameter values
tempPrior.otherInput = update_otherInput(tempPrior,hprstVec);
spatPrior.otherInput = update_otherInput(spatPrior,hprsxVec);

