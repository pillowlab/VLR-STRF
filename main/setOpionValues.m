function opts = setOpionValues(options);

opts = options;

% check if maxiter options are supplied
if isfield(opts,'maxiter')
    if ~isfield(opts.maxiter,'EM')
        opts.maxiter.EM = 25;
    end
    
    if ~isfield(opts.maxiter,'tempStep')
        opts.maxiter.tempStep = 20;
    end
    
    if ~isfield(opts.maxiter,'spatStep')
        opts.maxiter.spatStep = 20;
    end

    if ~isfield(opts.maxiter,'Mstep')
        opts.maxiter.Mstep = 20;
    end

else
    opts.maxiter.EM = 25;       % number of total EM iterations
    opts.maxiter.tempStep = 20; % number of temporal update iterations
    opts.maxiter.spatStep = 20; % number of spatial update iterations
    opts.maxiter.Mstep = 20;    % number of model params iterations
end

% check if verbose options
if ~isfield(opts,'verbose')
    opts.verbose = 1;
end

% check if reduced moment calculation options
if ~isfield(opts,'ReducedMoments')
    opts.ReducedMoments = 1;
end

