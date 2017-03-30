% LFM Toolbox for Matlab 
% Version 1.0, June 20. 2012
%
% Copyright (C) 2012 Jouni Hartikainen <jouni.hartikainen@aalto.fi>
%               2012 Simo S�rkk� <simo.sarkka@aalto.fi>
%               
% History:      
%   12.06.2012 Initial version
%
% This software is distributed under the GNU General Public
% Licence (version 3 or later); please refer to the file
% Licence.txt, included with the software, for details.
% 
%
% Inference with LTI SDEs
%   E_LTISDE   Marginal likelihood (ML) calculation for parameter inference
%   DE_LTISDE  Derivative of ML w.r.t. given parameters 
%   ES_LTISDE  Filtering and smoothing solutions 
%   
% Inference with non-linear SDEs
%   E_GF       ML with Gaussian filter
%   ES_GF      Filtering and smoothing solutions with Gaussian filter
%   E_SQRT_GF  ML with square root Gaussian filter
%   ES_SQRT_GF Filtering and smoothing solutions with sqrt Gaussian filter
%   E_SMC_SDE  ML with particle filter (returns also the particle approximations)
%   E_PDE_SDE  ML with brute force filter solving the FPK in a grid
%   DMP        Time derivatives of mean and covariance
%   DMPC       Time derivatives of mean, covariance and cross-covariance
%   DMA_SQRT   Time derivatives of mean and covariance (sqrt form)
%   DMAC_SQRT  Time derivatives of mean, cov. and cross cov. (sqrt form)
%
% Utility functions
%   LOGSUM     Log sum of logs
%   MYODE      Wrapper function needed in ODE moment integration
%   MA2VEC     Conversion functions used by sqrt non-linear filters
%   MAC2VEC    ...
%   VEC2MA     ...
%   VEC2MAC    ...
%   MP2VEC     Conversion functions used by non-linear filters
%   MPC2VEC    ...
%   VEC2MP     ...
%   VEC2MPC    ...
%
% /EKFUKF/     Stripped-down version of EKF/UKF toolbox 
%
% /MISC/.      Miscellaneous functions 
%     PRIOR_T    Student-t prior for parameters (taken from GPStuff package)
%     SRKS15V    Stochastic Runge-Kutta (strong order 1.5) for solving SDEs
%
% /MODELS/            Built-in LTI SDE model functions
%     GAUSSNCOV       Covariance matrix of Gaussian measurement noise
%     LFM1_MODEL      1st order Latent Force Model
%     LFM2_MODEL      2nd order Latent Force Model
%     MATERN_MODEL    Matern GP model
%     RESONATOR_MODEL Stochastic resonator model
%     SUM_MODEL       Model forming the signal as sums any LTI SDEs
%     WVEL_MODEL      Wiener velocity model
%
%
% /DEMOS/                 Demonstrations
%
%   /BALLISTIC/
%      BAllISTIC_DEMO     Demo of tracking a ballistic target on reentry
% 
%   /GINZBURG-LANDAU      Demo of Ginzburg-Landau double well model
%    
%   /GPR/             
%      GPR_DEMO           Demo of LTI SDE inference
%      GPR_SUM_DEMO       Demo of LTI SDE inference with sum model
% 
%   /LLFM/
%      LFM1_DEMO          Demo of 1st order linear LFM 
%      LFM2_DEMO          Demo of 2nd order linear LFM
%  
%   /TF/
%      TF_DEMO            Demo of 1st order non-linear LFM (TF model)
%
%   /VAN-DER-POL/
%      VDP_DEMO           Demo of Van der Pol oscillator model


