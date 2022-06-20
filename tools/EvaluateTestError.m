function [msetest, stdtest] = EvaluateTestError(Xtest,Ytest,k,dc);

filterResp = sameconv(Xtest,k) + dc;

msetest = mean((Ytest - filterResp).^2);
stdtest = std((Ytest - filterResp).^2);