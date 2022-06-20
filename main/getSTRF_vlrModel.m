function [mut,mux] = getSTRF_vlrModel(m)
% [mut,mux] = getSTRF_vlrModel(m)
% 
% Extract STRF as a matrix from model parameters

% build temporal Basis
hprstVec = getHyperParams(m.tempPrior); % get temporal parameters as a vector
Bkt = m.tempPrior.BasisFun(hprstVec,m.tempPrior); % time basis

% build spatial basis
hprsxVec = getHyperParams(m.spatPrior); % get spatial hyperparameters as a vector
Bkx = m.spatPrior.BasisFun(hprsxVec(2:end),m.spatPrior,[],0); % space basis without trimming

mut = Bkt*reshape(m.tempPost.Mean, [size(Bkt,2),m.strfRank]);
mux = Bkx*reshape(m.spatPost.Mean, [m.strfRank,size(Bkx,2)])';
