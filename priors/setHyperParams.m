function hprsStruct = setHyperParams(hyperparamVec,name)
% hprsStruct = setHyperParams(hyperparamVec)
%
% Extract prior hyperparameters as a vector
%
% Duncker & Pillow, 2018-2020
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch name
    
    case 'ASD' % Automatic Smootheness Determination prior
        % extract named hyperparameters from input
        if (length(hyperparamVec) == 2) % 1D case
            hprsStruct.rho = hyperparamVec(1);
            hprsStruct.space_len = hyperparamVec(2);
            
        elseif (length(hyperparamVec) == 3) % 2D case
            hprsStruct.rho = hyperparamVec(1);
            hprsStruct.space_len = hyperparamVec(2:3);
        else
            error('length of hyperparamVec must be 2 or 4 for ASD');
        end
        
    case 'ALD' % Automatic Locality Determination prior
        
        hprsStruct.rho = 1; % marginal variance
        hyperparamVec = hyperparamVec(2:end); % ignore rho entry
        
        if (length(hyperparamVec) == 4) % 1D case
                
            hprsStruct.space_mu = hyperparamVec(2); % mean
            hprsStruct.space_len = 1./hyperparamVec(1); % lengthscale
            hprsStruct.freq_mu = hyperparamVec(4); % mean
            hprsStruct.freq_len = 1./hyperparamVec(3); % lengthscale
            
        elseif (length(hyperparamVec) == 10) % 2D case
            
            hprsStruct.space_mu = hyperparamVec(4:5)'; % space mean
            hprsStruct.space_len = 1./hyperparamVec([1 3])'; % space lengthscale
            hprsStruct.space_corr = hyperparamVec(2); % space correlation
            hprsStruct.freq_mu = hyperparamVec(9:10)'; % freq mean
            hprsStruct.freq_len = 1./hyperparamVec([6 8])'; % freq lengthscale
            hprsStruct.freq_corr = hyperparamVec(7); % freq correlation

        else
            error('length of hyperparamVec must be 4 or 10 for ALD');
        end
        
    case 'TRD' % Temporal Recency Determination prior
        
        hprsStruct.rho = hyperparamVec(1);
        hprsStruct.len = hyperparamVec(2);
        hprsStruct.c = hyperparamVec(3);        
        
    case 'RR' % Ridge regression prior

        hprsStruct.rho = hyperparamVec(1);

end