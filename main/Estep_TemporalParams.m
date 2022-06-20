function [mut,Sigmat] = Estep_TemporalParams(XY,Xmu,rnk,mux,Bkt,Bkx,sigma,dc,Q1);
% perform E-step update

nt = size(Bkt,2);
nx = size(Bkx,2);

% === Compute Sigmat ====
D = zeros(rnk*nt,rnk*nt);
for i = 1:rnk
  D(1 + (i-1)*nt : nt + (i-1)*nt,1 + (i-1)*nt: nt +(i-1)*nt) = Bkt'*Q1(:,:,i,i)*Bkt;
  for j = i+1:rnk
      D(1 + (i-1)*nt:nt+(i-1)*nt,1 + (j-1)*nt:nt+(j-1)*nt) = Bkt'*Q1(:,:,i,j)*Bkt;
      D(1 + (j-1)*nt:nt+(j-1)*nt,1 + (i-1)*nt:nt+(i-1)*nt) = Bkt'*Q1(:,:,i,j)'*Bkt;
  end
end
Sigmat = (eye(rnk*nt) + 1/sigma^2*D)\eye(rnk*nt);


% === Compute mut ====
XYmom = XY-dc*Xmu; % reweight moment based on dc
XYproj = (Bkt'*XYmom*Bkx); % reshape and project onto temporal and spatial bases
krmx = kron(reshape(mux,[rnk,nx])',eye(nt));
mut = 1/sigma^2*Sigmat*(krmx'*XYproj(:));

end

