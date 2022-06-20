function kSTRF = get_estimate_from_output(struct,nn,rr,varargin)

if any(size(struct) ~= [1 1])
    kSTRF = struct(nn,rr).Basis.t*reshape(struct(nn,rr).PosteriorMean.t,...
        [size(struct(nn,rr).Basis.t,2),rr])*reshape(struct(nn,rr).PosteriorMean.x,[rr,size(struct(nn,rr).Basis.x,2)])*struct(nn,rr).Basis.x';
else
    kSTRF = struct.Basis.t*reshape(struct.PosteriorMean.t,...
        [size(struct.Basis.t,2),rr])*reshape(struct.PosteriorMean.x,[rr,size(struct.Basis.x,2)])*struct.Basis.x';
end
    
if nargin > 3
    if strcmp(varargin{1},'normalize')
        kSTRF = kSTRF./norm(kSTRF(:));
    end
end