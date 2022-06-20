function hprs = initialiseHprs_RR(varargin)
% hprs = initialiseHprs_RR(k0,RFdims)
%
% Initialize hyperperparameters of Ridge Regression (RR)
% model for 1D or 2D spatial RF 
%
% INPUT:
%     k0 [nt x nkx] - STA or other initial filter estimate (as matrix)
%  RFdims [nx x ny ] - size of spatial RF
%
% OUTPUT:
%  hprsSpace - initial ALD hyperparameters, which are:
%
%     hprs.rho = 1 (marginal variance default)

hprs.rho = 1;  % initialize marginal variance param
