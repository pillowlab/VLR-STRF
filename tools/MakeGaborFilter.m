function kx = MakeGaborFilter(xx,yy,gamma,lambda,sigma,theta,phi,nkx1,nkx2);

xx1 = xx.*cos(theta) + yy.*sin(theta);
xx2 = -xx.*sin(theta) + yy.*sin(theta);
gabor = exp(- pi.*(xx1.^2 + gamma^2*xx2.^2)*(2*sigma^2)).*cos(2*pi*(xx1/lambda)+phi);

kx = (reshape(gabor,[nkx1 nkx2]));
kx = -kx./norm(kx(:));
