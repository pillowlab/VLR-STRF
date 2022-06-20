function Msum = mkDiagSummingMtx(n)
% M = mkDiagSummingMtx(n)
%
% Make sparse matrix M that will sum up the diagonals of a square (n x n) 
% matrix via M*A(:);


Msum = sparse([],[],[],n^2,2*n-1,n^2);
for jj = 1:2*n-1
    A = spdiags(ones(min(jj,n),1),jj-n,n,n);
    Msum(:,jj) = A(:);
end
Msum = Msum';
