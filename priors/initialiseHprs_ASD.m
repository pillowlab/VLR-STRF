function hprs = initialiseHprs_ASD(k0,RFdims)
% hprs = initialiseHprs_ASD(k0,RFdims)
%
% Initialize hyperperparameters of automatic smoothness determination (ASD)
% model for 1D or 2D spatial RF 
%
% INPUT:
%     k0 [nt x nkx]  - STA or other initial filter estimate (as matrix)
%  RFdims [nx x ny ] - size of spatial RF
%
% OUTPUT:
%  hprsSpace - initial ASD hyperparameters, which are:
%
%     hprs.rho = 1 (marginal variance default)
%     hprs.space_len = Spatial length scale 
%     hprs.space_corr = correlation for 2D RF


hprs.rho = 1;  % initialize marginal variance param

% Do an SVD of full STA to get spatial weighting function
spatialInitMethod = 2; % 1 for SVD; 2 for spatially summed STA
switch(spatialInitMethod)
    case 1 % Method 1: use SVD of smoothed STA
        [~,~,v] = svd(gsmooth(k0',2)');
        xwts = abs(v(:,1)).^2;  % use squared coeffs (or abs if desired)
    case 2 % Method 2: use spatially summed STA
        xwts = sum(k0.^2,1)';
end
xwts = xwts-median(xwts(:));  % subtract off median (to remove impact of spurious coeffs near zero)
xwts(xwts<0) = 0; % eliminate neg values
xwts = xwts/sum(xwts);  % normalize weights

if any(RFdims == 1) % 1D spatial RF

    % =====================================================
    % === Initialize spatial-domain hyperparameters =======

    nx = prod(RFdims);
    xx = 1:nx; % spatial grid

    % Compute spatial mean and length scale
    space_mu = xx*xwts; % spatial mean
    hprs.space_len = sqrt((xx-space_mu).^2*xwts); 
    
elseif all(RFdims > 1) % 2D spatial RF

    xwts = reshape(xwts,RFdims);
    
    % === Initialize spatial-domain hyperparameters =======
    xx1 = 1:RFdims(1); % grid for 1st spatial dimension
    xx2 = 1:RFdims(2); % grid for 2nd spatial dimension

    % Compute spatial mean and length scale
    x_mu1 = sum(xx1*xwts); % spatial mean 1
    x_mu2 = sum(xx2*xwts'); % spatial mean 2   
    x_len1 = sqrt(sum((xx1-x_mu1).^2*xwts)); % spatial inverse length scale 1
    x_len2 = sqrt(sum((xx2-x_mu2).^2*xwts')); % spatial inverse length scale 2   

    % Assemble hyperparams
    hprs.space_len = [x_len1, x_len2]; % space lengthscales

   
end

