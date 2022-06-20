function [MSE,Rsq] = evaluate_EstVsTrue(kest,ktrue);

mse = sum((kest(:) - ktrue(:)).^2);
r2 = 1 - mse./sum(ktrue(:).^2);

MSE = mse;
Rsq = r2;

