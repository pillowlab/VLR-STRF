function kSTA = computeSTA(Y,Stim,nkt);

kSTA = simpleRevcorr(Stim,Y,nkt);
kSTA = kSTA./norm(kSTA); % normalize