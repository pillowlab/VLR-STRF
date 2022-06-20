function vec = getHyperParams(priorstruct)
% vec = getHyperParams(priorstruct)
%
% Extract prior hyperparameters as a vector
%
% Input: 
%    priorstruct - prior structure with fields:
%     .name ('ASD', 'ALD', 'TRD', 'Ridge') - specifies kind of prior
%     .hprs - struct of hyperparameters for the corresponding prior
%
% Output:
%    vec - vector of hyperparameters for associated prior
%
% Duncker & Pillow, 2018-2020
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hprs = priorstruct.hprs;

switch priorstruct.name
    
    case 'ASD' % Automatic Smootheness Determination prior
        % extract named hyperparameters from input
        xscale = hprs.rho;

        if (length(hprs.space_len) == 1) % 1D case
            xlen = hprs.space_len;
     
            % assemble into vector
            vec = [xscale; xlen];
            
        elseif (length(hprs.space_len) == 2) % 2D case
             xlen = hprs.space_len;  % space lengthscales
             
             % assemble into vector
             vec = [xscale; xlen(1); xlen(2)];
        end
        
    case 'ALD' % Automatic Locality Determination prior
        xscale = hprs.rho;

        if (length(hprs.space_mu) == 1) % 1D case

            xmu = hprs.space_mu; % mean
            xlen = hprs.space_len; % inverse lengthscale
            fmu = hprs.freq_mu; % mean
            flen = hprs.freq_len; % inverse lengthscale

            % assemble into vector
            vec = [xscale; 1./xlen;xmu;1./flen;fmu]; 

        elseif (length(hprs.space_mu) == 2) % 2D case
            
            xmu  = hprs.space_mu'; % space mean
            xlen = hprs.space_len;  % space lengthscales
            xcr  = hprs.space_corr; % space correlation
            fmu  = hprs.freq_mu'; % freq mean
            flen = hprs.freq_len; % freq lengthscales
            fcr  = hprs.freq_corr; % freq correlation           
            
            % assemble into vector
            try
                vec = [xscale; 1./xlen(1); xcr; 1./xlen(2); xmu; ...
                    1./flen(1); fcr; 1./flen(2); fmu]; 
            catch
                keyboard;
            end
        else
            error('length of hprs.space_mu vector must be 1 or 2'); 
        end
        
        
    case 'TRD' % Temporal Recency Determination prior
        
        vec = [hprs.rho; hprs.len; hprs.c];
        
    case 'RR' % Ridge regression prior
       
        vec = [hprs.rho];
end