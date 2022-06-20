# VLR - variational inference for low-rank STRFs

This repository contains a Matlab toolbox for inferring approximate posterior estimates of the temporal and spatial components of a spatio-temporal receptive field (STRF) with low-rank structure. To cite this code, please refer to our paper
```
@ARTICLE{duncker+pillow:2022,
  author = {Duncker, Lea and Pillow, Jonathan W.},
  title =  {Scalable variational inference for low-rank spatio-temporal receptive fields},
  journal = {},
  year    = {2022},
}
```

## Code Usage
### Data Format
The toolbox expects data input in the form of a response vector `Y` and a stimulus `X`. For `N` samples, `Y` is an `N x 1` vector and `X` is an `N x D` matrix, where `D` is the total number of stimulus coeffients. The spatial receptive field dimensions will need to match those of the stimulus. For example, for an image with `D=800` total coefficients and dimensions `nkx1 = 20` and `nkx2 = 40`, the spatial receptive field would have dimensions `RFdims = [nkx1 nkx2]`.

The code also requires specifying the length of a single time bin (`dtbin`), the total extent of the temporal receptive field both in terms of the number of coefficients (`nkt`) and in units of time (`tmax`), and a minimum temporal lengthscale (`minlen_t` in units of time).

### Specifiying Priors
Since the code implements maximum aposterior estimation for receptive fields with low-rank structure, the first step is to define a covariance function and specify which prior to use for the temporal and spatial components of the STRF. The toolbox contains a number of commonly used priors for hierarchical receptive field estimation, which can be combined in a flexible way. The available covariance functions, their names in the toolbox, and implied prior assumptions are summarized below

prior covariance | name | receptive field component | assumptions about receptive field | Reference
---------------- | ------------ | --------- | ---------------------| ---------
Automatic Smoothness Determination | ASD | spatial, temporal | receptive field varies smoothly over its coefficients |[2]
Autmatic Locality Determination | ALD | spatial | receptive field varies smoothly and its non-zero coefficeints are localized | [3]
Temporal Recency Determination | TRD | temporal | receptive field smoothness increases with extent in time (non-stationary smoothness)| [1]
Ridge Regression | RR | spatial, temporal | uncorrelated RF coefficients |

To build an ALD prior for the spatial component of the receptive field, and a TRD prior for the temporal component, you can call the following code

```matlab
% build prior for spatial receptive field
RFdims = [nkx1 nkx2];
spatPrior = build_vlrPrior('ALD',RFdims);

% build prior for temporal receptive field
minlen_t = 5*dtbin;  % minimum temporal lengthscale (in s)
tempPrior = build_vlrPrior('TRD',nkt,minlen_t,tmax);
```
To initialise the hyperparameters of the priors based on an estimate of the spike-triggered-average `kSTA` of dimensions  `nkt x D` , call

```matlab
[tempPrior, spatPrior] = initialiseHprs_vlrPriors(kSTA,tempPrior,spatPrior);
```

### Building the model

After having specified the priors for each receptive field component, a model for a spatio-temporal receptive field with rank `rnk` can be built by calling
```matlab
opts = []; % use default options
rnk = 2; % specify rank of STRF
% build model structure
mdl = build_vlrModel(Y,X,rnk,spatPrior,tempPrior,opts);
```
This also computes the sufficient statistics of the input data, so it might take longer for large sample sizes.

### Optimizing the model
The model structure `mdl` contains initial estimates for the hyperparameters and receptive field components. In order to find the maximum aposteriori estimate, run variational Expectation Maximisation (vEM) by calling
```matlab
mdl = fit_vlrModel(mdl);
```
###  Final STRF estimate
Finally, one can obtain a full-sized STRF estimate `kMAP` from the optimized model structure by calling
```matlab
[mutHat,muxHat]  = getSTRF_vlrModel(mdl);
kMAP = mutHat*muxHat';
```
`mutHat` and `muxHat` are the orthogonalised temporal and spatial receptive field components, respectively.

Further examples on how to use the code are provided in the demos folder.

## References:
[1] L Duncker, JW Pillow. Scalable variational inference for low-rank spatio-temporal receptive fields. In prep.

[2] M Sahani, and J F Linden. Evidence optimization techniques for estimating stimulus-response functions. Advances in neural information processing systems. 2003.

[3] M Park and J W Pillow, Receptive field inference with localized priors. PLoS computational biology, 7(10), 2011, p.e1002219.

