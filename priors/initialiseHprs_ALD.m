function hprs = initialiseHprs_ALD(k0,RFdims)
% hprs = initialiseHprs_ALD(k0,spatPrior)
%
% Initialize hyperperparameters of automatic locality determination (ALD)
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
%     hprs.space_mu = mean spatial location of RF
%     hprs.space_len = Spatial length scale 
%     hprs.space_corr = correlation (2D only)
%     hprs.freq_mu = Fourier-domain mean location
%     hprs.freq_len = Fourier length scale 
%     hprs.freq_corr = correlation (2D only)

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
    hprs.space_mu = xx*xwts; % spatial mean
    hprs.space_len = sqrt((xx-hprs.space_mu).^2*xwts); % spatial inverse length scale

    % % compare shapes, if desired
    % px = normpdf(xx,x_mu,x_len);
    % plot(xx,px/sum(px),xx,xwts);
    
    % =====================================================
    % === Initialize Fourier-domain hyperparameters =======    

    ww = -ceil((nx-1)/2):floor((nx-1)/2); % Fourier frequencies
    fwts = fftshift(abs(fft(xwts)).^2);  % Fourier weights
    fwts = fwts-median(fwts(:)); % subtract median
    fwts(fwts<0) = 0; % remove neg values
    fwts = fwts/sum(fwts); % normalized Fourier weights 
    
    hprs.freq_mu = 0.1; % mean (initialize close to zero, implying smoothness)
    hprs.freq_len = sqrt((ww-hprs.freq_mu).^2*fwts);

    % % compare shapes, if desired
    % pw = normpdf(ww,hprs.freq_mu,hprs.freq_len);
    % plot(ww,pw/sum(pw),ww,fwts);
    
    % Assemble hyperparams

    
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

    % compare shapes, if desired
%     [x1,x2] = ndgrid(xx1,xx2); z = [x1(:), x2(:)];
%     px = mvnpdf(z,[x_mu1 x_mu2], (diag([x_len1,x_len2])));
%     subplot(221); imagesc(xwts); title('spatial');
%     subplot(222); imagesc(reshape(px,RFdims));
     
    
    % =====================================================
    % === Initialize Fourier-domain hyperparameters =======    
    ww1 = -ceil((RFdims(1)-1)/2):floor((RFdims(1)-1)/2); % Fourier frequencies 1
    ww2 = -ceil((RFdims(2)-1)/2):floor((RFdims(2)-1)/2); % Fourier frequencies 2    
    fwts = fftshift(abs(fft2(xwts)).^2);  
    fwts = fwts-median(fwts(:));
    fwts(fwts<0) = 0;
    fwts = fwts/sum(fwts(:)); % Fourier weights
    w_mu1 = 0.11; % mean 1 (initialize close to zero, implying smoothness)
    w_mu2 = 0.12; % mean 2
    w_len1 = sqrt(sum((ww1-w_mu1).^2*fwts));
    w_len2 = sqrt(sum((ww2-w_mu2).^2*fwts'));    

    % % compare shapes, if desired
    [w1,w2] = ndgrid(ww1,ww2); zw = [w1(:), w2(:)];
    pw = mvnpdf(zw,[w_mu1 w_mu2], inv(diag([w_len1,w_len2])));
%     subplot(223); imagesc(sqrt(fwts)); title('Fourier');
%     subplot(224); imagesc(reshape(pw,RFdims));
%     
    % Assemble hyperparams
    hprs.space_mu = [x_mu1, x_mu2]; % space mean
    hprs.space_len = [x_len1, x_len2]; % space lengthscales
    hprs.space_corr = 0.01; % correlation
    hprs.freq_mu = [w_mu1 w_mu2]; % freq mean
    hprs.freq_len = [w_len1, w_len2]; % freq lengthscales
    hprs.freq_corr = 0.01; % correlation
   
end

