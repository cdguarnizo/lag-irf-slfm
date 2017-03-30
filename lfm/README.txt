Latent Force Model Toolbox (LFMT) for Matlab

Authors: Jouni Hartikainen <jouni.hartikainen@aalto.fi>
         Simo Särkkä       <simo.sarkka@aalto.fi>

INTRODUCTION

LFMT is a collection of routines that can be used to construct linear
and non-linear latent force models (LFMs) in state-space form, and
infer the state and parameters in these models with various Kalman
filtering and smoothing type of methods. In a more general sense, the
toolbox allows the user to construct models that are described with
either linear or non-linear stochastic differential equations (SDEs)
that are observed discretely in time. Since the estimation (that is,
filtering and smoothing) solutions for linear and non-linear cases
differ considerably from each other, these inference routines function
differently for both cases. However, the linear models can easily be
used as components of non-linear models, which is also demonstrated
via several examples.

Currently, the toolbox has the following features:

Linear models:
  - Assumed model class: linear time-invariant (LTI) SDEs of form
   
    dx(t)/dt = F x(t) + L w(t),

    where x(t) is the state, F a constant drift matrix and L a
    constant dispersion matrix, and w(t) a white noise process has
    spectral density Qc.
  - Built-in models: State-space Matern GP, 1st and 2nd order LFMs,
    stochastic resonator model, Wiener velocity model, sum model
    combining any of the above
  - State inference: Kalman filter and smoother 
  - Parameter inference: Optimization of marginal likelihood (with
    gradients) and adaptive MCMC (Robust Metropolis, RAM)
  - Several demonstrations

Non-Linear models
  - Assumed model class: non-linear SDEs of form

    dx(t)/dt = f(x(t),t) + L(x(t),t) w(t),

    where f and L are non-linear drift and dispersion matrices that
    can depend on x as well as t, and w(t) a white noise process
    similarly as in the linear case.
  - Built-in models: Transcription factor model (1st order non-linear LFM),
    ballistic re-entry target model with range measurements,
    Ginzburg-Landau double well model, Forced Van der Pol oscillator model
  - State inference: Continuous-discrete Gaussian filter and smoother
    (in both covariance and square-root forms with various numerical
    schemes for approximating the needed integrals), particle filter
    with stochastic Runge-Kutta
  - Parameter inference: Optimization of marginal likelihood (without
    gradients) and adaptive MCMC (RAM)
  - Several demonstrations

In both linear and non-linear cases the measurement models are of the
form 

   y_k = h(x(t_k),k) + r_k,
   
   where r_k is N(0,R_k) distributed noise.

For more details, see the papers:

Jouni Hartikainen, Mari Seppänen and Simo Särkkä (2012). State-Space
Inference for Non-Linear Latent Force Models with Application to
Satellite Orbit Prediction. ICML 2012.

Jouni Hartikainen and Simo Särkkä (2011). Sequential Inference for
Latent Force Models. UAI 2011.

Jouni Hartikainen and Simo Särkkä (2010). Kalman Filtering and
Smoothing Solutions to Temporal Gaussian Process Regression
Models. MLSP 2010.

LICENCE

This software is distributed under the GNU General Public Licence
(version 3 or later); please refer to the file Licence.txt, included
with the software, for details.

BRIEF USER GUIDE 

The main functionality of the toolbox resides in inference routines,
which can be used to compute the filtering and smoothing solutions
to the model classes (linear and non-linear SDEs) assumed by the
toolbox as well as the marginal likelihood used for parameter estimation.

For linear SDEs the toolbox the following inference functions:
    E_ltisde  - Marginal likelihood (ML) calculation for parameter inference
    DE_ltisde - Derivative of ML w.r.t. given parameters 
    ES_ltisde - Filtering and smoothing solutions 

For non-linear SDEs the corresponding functions are
    E_gf       - ML with Gaussian filter
    ES_gf      - Filtering and smoothing solutions with Gaussian filter
    E_sqrt_gf  - ML with square root Gaussian filter
    ES_sqrt_gf - Filtering and smoothing solutions with sqrt Gaussian filter
    E_smc_sde  - ML with particle filter (returns also the particle approximations)
    E_pde_sde  - ML with a brute force filter solving the FPK in a grid

The interface to these functions are of form

output = function(theta,param)

where theta is a parameter parameter vector and param a structure
containing all the relevant information about the model, data and
inference settings. The contents of output argument depend on the
actual function in use. The details about the output and input
arguments are documented in the actual code files.

The models are constructed a bit differently in linear and non-linear
cases. 

In linear cases the models constructed with single functions of form

model = model_function(theta,param,calc_grad)

where model is a structure containing all the model parameters needed
by the inference routines (that is, matrices F, L and Qc), theta a
vector of free parameters, param a structure for passing relevant
information about the model for the particular model function (for
example, which of the components of theta are fixed during parameter
estimation, what is the degree of a variable size model etc.). The
last input argument calc_grad tells the function whether to calculate
the gradients of the model parameters to model structure.

The non-linear models are divided to three separate functions:
model_f, model_q and model_set_params. 

The model_f function has the interface

f = model_f(x,t,f_param),

where f is the deterministic part of the time derivative of state x,
t is the time instance and f_param a parameter structure specific to
actual model function.

Similarly, model_q has the interface

q = model_q(x,t,q_param),

where q is the term L(x(t),t) G, where G is the Cholesky factor of
Qc (the spectral density of process noise w(t)), and q_param a
parameter structure specific to model_q.

The last function model_set_params has the interface

[param f_param q_param] = model_set_params(param,f_param,q_param,theta),

and is used to setup the modelstructures f_param and q_param using the
vector of free parameters theta with interface. The function also can
setup the general parameter structure param, which contains, for example,
the prior mean and and covariance.

See the demonstrations how to setup the needed parameter structures,
how the construct different models and how to perform state and
parameter inference. Feel free to modify the existing files and
experiment on your own!






 