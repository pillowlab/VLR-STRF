function X = makeDesignMat(Stim,Bkt,Bkx)
% makes stimulus design matrix projected onto temporal and spatial bases
%
% ---- Set up filter and stim processing params ------------------- 
N = size(Stim,1); % size of stimulus samples x # pixels
nkx = size(Bkx,2); % # params per spatial vector
nkt = size(Bkt,2); % # time params per stim pixel (# t params)
ncols = nkx*nkt; % number of columns in design matrix 
% ---- Convolve stimulus with spatial and temporal bases -----

% ---- Convolve stimulus with spatial and temporal bases -----
xfltStim = Stim*Bkx; % stimulus filtered with spatial basis
X = zeros(N,ncols); %
for i = 1:nkx
    for j = 1:nkt
        X(:,(i-1)*nkt+j) = sameconv(xfltStim(:,i),Bkt(:,j));
    end
end
