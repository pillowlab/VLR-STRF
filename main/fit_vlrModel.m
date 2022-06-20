function m = fit_vlrModel(m)
% function  m = fit_vlrModel(m);
%
% function to perform variational inference and hyperparameter learning for
% low-rank STRFs
% 
% inputs:
% ----------
%       m          --- model structure built using build_vlrPrior.m
% 
% outputs:
% ----------
%       m           --- updated model structure after fitting
%
% See also: build_vlrPrior.m
%
% Duncker & Pillow, 2018-2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m.opts.abstol = 1e-02;  % stop if EM progresses this much only
tstart = tic;

if m.opts.verbose 
       fprintf('%3s\t%10s\t%10s\n', 'iter', 'objective', 'increase');
end

% run variational EM
for i = 1:m.opts.maxiter.EM
    
    % update spatial parameters: prior hyperparameters, variational parameters
    m = UpdateSpatialParams(m);
    
    % update temporal parameters: prior hyperparameters, variational parameters
    m = UpdateTemporalParams(m);
    
    % update model parameters: noise variance and constant offset
    m = updateModelParams(m);
    
    % compute value of free Energy
    m.history.VarFreeEnergy(i) = variationalFreeEnergy(m);

    % produce some output
    if m.opts.verbose
        if i > 1
            fprintf('%3d\t%10.4f\t%10.4f\n', i, m.history.VarFreeEnergy(i), m.history.VarFreeEnergy(i) - m.history.VarFreeEnergy(i-1));
        else
            fprintf('%3d\t%10.4f\n', i, m.history.VarFreeEnergy(i));
        end
    end
    
    % check convergence in Free Energy
    if i > 1 && abs(m.history.VarFreeEnergy(i) - m.history.VarFreeEnergy(i-1)) < m.opts.abstol
        break;
    end
    
end
% report elapsed time
if m.opts.verbose 
    toc(tstart)
end
% report warning if EM hasn't converged
if i == m.opts.maxiter.EM && m.opts.verbose 
    warning('Max # of iterations reached: variational EM did not yet converge to maximum')
    fprintf('\n Increase ''maxiter'' if higher optimum desired\n\n');
end

