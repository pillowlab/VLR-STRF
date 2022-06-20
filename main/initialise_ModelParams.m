function [ss,bb] = initialise_ModelParams(Y);
% function to initialise model parameters
ss = std(Y);
bb = mean(Y);
