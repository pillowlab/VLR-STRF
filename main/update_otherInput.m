function otherInput = update_otherInput(kern,hprs)

switch kern.name
    
    case 'ASD' % Automatic Smootheness Determination prior
        
        otherInput = [];
        
    case 'ALD' % Automatic Locality Determination prior

        otherInput = [];
       
    case 'TRD' % Temporal Recency Determination prior

        otherInput = kern.otherInput;
        otherInput.Tcirc = kern.otherInput.tmax + 3*hprs(2);
        
    case 'RR' % Ridge regression prior

        otherInput = [];
end