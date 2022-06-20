function [mux,Sigmax] = Estep_SpatialParams(XY,Xmu,rnk,mut,Bkt,Bkx,sigma,dc,Q1,BasisInfo);
% perform E-step update for spatial parameters

idx = BasisInfo.idx;

nt = size(Bkt,2);
nx = size(Bkx,2);

% === Compute Sigmax =====
P = commutation(rnk,nx);
D = zeros(rnk*nx,rnk*nx);
for i = 1:rnk
  D(1 + (i-1)*nx : nx + (i-1)*nx,1 + (i-1)*nx: nx +(i-1)*nx) = Bkx'*Q1(idx,idx,i,i)*Bkx;
  for j = i+1:rnk
      D(1 + (i-1)*nx:nx+(i-1)*nx,1 + (j-1)*nx:nx+(j-1)*nx) = Bkx'*Q1(idx,idx,j,i)'*Bkx;
      D(1 + (j-1)*nx:nx+(j-1)*nx,1 + (i-1)*nx:nx+(i-1)*nx) = Bkx'*Q1(idx,idx,j,i)*Bkx;
  end
end
Sigmax = (eye(rnk*nx) + P'*(1/sigma^2*D)*P)\eye(rnk*nx);

% === Compute mux ====
XYmom = XY(:,idx)-dc*Xmu(:,idx); % reweight moment based on dc
XYproj = Bkt'*XYmom*Bkx; % reshape and project onto temporal and spatial bases
krmt = repdiag(reshape(mut,[nt,rnk])',nx); % kronecker matrix
mux = 1/sigma^2*Sigmax*(krmt*XYproj(:));

end
