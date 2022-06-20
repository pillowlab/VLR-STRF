function kern = build_vlrPrior(name,nk,varargin)
% function kern = build_vlrPrior(name,nk,hprs,otherInput{:});
%
% function to initialise structure for variational low-rank strf prior
% 
% core inputs:
% ----------
%       name         --- {'ASD','ALD','TRD','RR'}
%       nk           --- [nk1 nk2] vector with RF dimensionality
%       hprs         --- initial hyperparameter vector, see kernel functions
%                        for detials on order 
% other inputs:
% ----------
%       minlens      --- for TRD prior only: minimum lengthscale
%       tmax         --- for TRD prior only: RF length in units of time
%
% outputs:
% ----------
%       kern        --- model structure to be passed into build_vlrModel
%                       function
%
% See also: build_vlrModel.m, fit_vlrModel.m
%
% Duncker & Pillow, 2018-2020
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch name
    
    case 'ASD' % Automatic Smootheness Determination prior        
        kern.name = name;
        kern.dims = nk;
        
        if any(nk == 1)
            kern.nhprs = 2; % [scaling, smoothness]
            kern.LowerOptimBounds = [1e-05; 1e-05];
            kern.UpperOptimBounds = [Inf; Inf];

        else
            kern.nhprs = 3; % [scaling, smooth1, smooth2]
            kern.LowerOptimBounds = [1e-05; 1e-05; 1e-05]; 
            kern.UpperOptimBounds = [Inf; Inf;Inf];
        end
        kern.BasisFun = @PriorBasis_ASD;
        kern.hprs = [];

    case 'ALD' % Automatic Locality Determination prior
        kern.name = name;
        kern.dims = nk;
        % not needed if minFunc used
        if any(nk == 1)
            kern.nhprs = 5; % includes scaling but gets fixed for spatial rf
            kern.LowerOptimBounds = [1e-05;1e-06;-1;1e-06;-Inf];
            kern.UpperOptimBounds = [Inf; Inf; max(nk)+1;Inf;0.5*max(nk)+1];
        else
            kern.nhprs = 11;
            kern.LowerOptimBounds = [1e-05; 1/nk(1);-1;1/nk(2);-1;-1;1e-06;1e-06;1e-06;-1;-1];
            kern.UpperOptimBounds = [Inf; Inf;1;10; nk(1)+1; nk(2)+1;Inf;Inf;Inf;0.5*nk(1)+1;0.5*nk(2)+1];
        end
        kern.BasisFun = @PriorBasis_ALD;
        kern.hprs = [];
        kern.otherInput = [];
        
    case 'TRD' % Temporal Recency Determination prior
        kern.name = name;
        kern.dims = nk;
        kern.nhprs = 3; % [scaling; smoothness; warping]
        kern.otherInput.minlen = varargin{1}; % minimum length scale
        kern.otherInput.Tcirc  = varargin{2} + 4*varargin{1}; % location of circular boundary
        kern.otherInput.tmax  = varargin{2}; % maximum time bin (s)
        kern.LowerOptimBounds = [1e-06;varargin{1};-5];
        kern.UpperOptimBounds = [Inf;Inf;5];
        kern.BasisFun = @PriorBasis_TRD;
        kern.hprs = [];

    case 'RR' % Ridge regression prior
        kern.name = name;
        kern.dims = nk;
        kern.nhprs = 1; %[scaling]
        kern.LowerOptimBounds = 1e-05;
        kern.UpperOptimBounds = Inf;
        kern.BasisFun = @PriorBasis_RR;
        kern.hprs = [];        
        kern.otherInput = [];
end