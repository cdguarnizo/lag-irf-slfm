% Demonstration of Impulse response estimation of a MIMO LTI system with 
% Kalman filtering and smoothing.
% 
% The model for i = 1,...,N outputs is
% 
% y_i(t) = \int h(t-tau) \sum_{r=1}^R u_r(tau) dtau,
% 
% where the input functions u_r(t) have GP priors (augmented as a part of 
% the joint state-space model). The impulse response h(t) is approximated 
% by Laguerre orthonormal functions as
%
% h(t) = \sum_{n=1}^N c_n l_n(t) 
%
% Copyright (C) 2016 Cristian Guarnizo
% Based on codes from Simo Sarkka, Jounin Hartikainen and Arno Solin
%
% This software is distributed under the GNU General Public 
% Licence (version 3 or later); please refer to the file 
% Licence.txt, included with the software, for details.

clc
clear
close all
warning off
%add to the path lfm and Laguerre model folders
addpath(genpath('../lfm'),'../model','../sde_funcs');

load('../datasets/data_ex2.mat')
Y = [y;uq];

% Number of time steps in the time series
steps = length(t);
mind = true(1,steps);

% Number of observations to be generated
ny = steps;

% Time step size
t_min = min(t);
t_max = max(t);
dt = t(2)-t(1);
tgrid = linspace(t_min,t_max,steps);

%% Construction of LFM

% Should the parameters be optimized
do_optim = 1;

% Number of outputs 
N = 2;
% Number of coefficients
D = 10;
% Prior covariance for the outputs 
P0x = 1e-3*eye(N*D);

% Use ki variable to choose the model:
%
% ki = 1: LFM with Matern for forces
% ki = 2: LFM with Matern + resonator for forces 
ki = 1;
if ki == 1           
    % Matern
    nlf = 2;

    % Parameters for the model function    
    param = struct;
    param.N = N;
    param.R = nlf;
    param.D = D;
    param.P0x = P0x;
    param.model_funcs  = {@matern_model @matern_model};
   
    % Matern parameters
    lengthScale = 2;
    sigma       = 1;
    lfm_param1 = struct;
    lfm_param1.p = 2;
    lfm_param1.fixp_ind = 2;  % Second parameter (magnitude) is fixed to value 1
    lfm_param1.fixp_val = sigma;
    param.model_params = {lfm_param1 lfm_param1};
    theta_lf = [1;2];
    % Initial guess for the parameters
    w0_lf = zeros(size(theta_lf));
elseif ki == 2    
    % Matern + resonator
    nlf = 2;
   
    % Parameters for the model function
    param = struct;
    param.N = N;
    param.R = nlf;
    param.P0x = P0x;
    
    % Matern parameters
    lengthScale = .5;
    sigma1      = 1;
    lfm_param1 = struct;
    lfm_param1.p = 2;
    lfm_param1.fixp_ind = 2;  % Fix the second parameter (magnitude)
    lfm_param1.fixp_val = 1;
    
    % Resonator parameters
    omega       = 0.5;
    sigma2      = 0.0001;
    lfm_param2 = struct;
    lfm_param2.nc = 2;
    lfm_param2.fixp_ind = 2;  % Fix the second parameter (noise std)
    lfm_param2.fixp_val = .0001;
    
    param.model_funcs  = {@matern_model @resonator_model};
    param.model_params = {lfm_param1 lfm_param2};
    theta_lf  = [lengthScale;omega];
    
    % Initial guess for the parameters
    w0_lf = zeros(size(theta_lf));
    w0_lf(2) = -3;
end
param.incInput = true;

Rtr = diag((0.01*var(Y')));

ind_t = tgrid>2. & tgrid<3.;
Y(2,ind_t) = NaN;
ind_t = tgrid>3. & tgrid<4.;
Y(3,ind_t) = NaN;

% Randomize the ODE parameters
%Ct = rand(N,1);     %gamma
%Kt = rand(N,D);    %coefficient
%St = ones(N,nlf); %sensitivities

w0 =randn(N + D*N,1);
w0 = [w0;w0_lf];

%Y(1,80:120) = NaN;
n_func = @gaussncov;
n_param = {};
n_param{1} = N;
n_param{2} = length(w0)+1:length(w0)+N;
% Uncomment these to optimize the measurement noise parameters
%theta = [theta;sqrt(diag(Rtr))];
%w0 = [w0;-1*ones(size(H,1),1)];

model_func  = @laguerre_model;
model_param = param; 

%theta_prior = prior_t('init');
%theta_prior.s2 = .3;
ltr_ind = [1:N,D*N+N+1:length(w0)];

e_param = struct;
e_param.Y           = Y;
e_param.mind        = mind;
e_param.model_func  = model_func;
e_param.model_param = param;
e_param.R           = Rtr;
e_param.H           = [];
e_param.n_param     = [];
% Uncomment these to optimize the measurement noise parameters
% e_param.R           = n_func;
% e_param.n_param     = n_param;

e_param.dt          = dt;
e_param.ltr_ind     = ltr_ind;

% Prior for thetas
theta_prior = cell(1,length(w0));

for i = 1:length(w0)
    theta_prior{i} = prior_t;
    %theta_prior{i}.s2 = .3;
end
e_param.e_prior = theta_prior;

%% MAP optimization of parameters 
% Note: the parameter posterior is usually highly non-Gaussian and
% multimodal, so optimization might get stuck to a local mode and even
% fail numerically, depending on simulation and model settings.

% Function handles to energy function and its derivative

% With fixed noise variance
e_func = @(w) E_lti_sde(w,e_param);
eg_func = @(w) DE_lti_sde(w,e_param);

mydeal = @(varargin)varargin{1:nargout};

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-3);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');

if do_optim == 1
    w_opt = fminunc(@(ww) mydeal(e_func(ww), eg_func(ww)), w0', opt);
else    
    w_opt = log(theta)';
end

theta_opt = w_opt';
theta_opt(ltr_ind) = exp(w_opt(ltr_ind)');

% load ex2_irf.mat
% w_opt = theta_opt;
% w_opt(ltr_ind) = log(theta_opt(ltr_ind));


theta_opt

% Interpolation to a finer grid
isteps = 5;
tgrid2 = linspace(t_min,t_max,steps*isteps);
e_param2 = e_param;
e_param2.isteps = isteps;
[E2,MM2,PP2,MS2,PS2] = ES_lti_sde(w_opt,e_param2);

%% Plotting estimates of output signals
color1 = [0.7 0.7 0.7];
color2 = [1 0 0];
color3 = [0 1 0];
xx = tgrid'+dt;
xx2 = tgrid2'+dt./isteps;

model = feval(model_func,theta_opt,param,0);

H = model.H;

nisteps = steps*isteps;
MS2fi = H*MS2;

PS2fi = zeros(N,nisteps);

for i1 = 1:N
    for i = 1:size(PS2,3)
        PS2fi(i1,i) = H(i1,:)*PS2(:,:,i)*H(i1,:)';
    end
end

for i = 1:N
    figure(i); clf;
    fill([xx2' fliplr(xx2')], [(MS2fi(i,:)+1.96*sqrt(PS2fi(i,:))) ...
        fliplr((MS2fi(i,:)-1.96*sqrt(PS2fi(i,:))))], color1, 'edgecolor',color1);
    hold on
    
    h1=plot(xx2,MS2fi(i,:),'--k','LineWidth',1);
    h2 = plot(xx(mind==1),Y(i,:),'.k','LineWidth',1);
    flag = isnan(Y(i,:));
    if any(flag),
        h3 = plot(xx(flag),y(i,flag),'.r','LineWidth',1);
    end
    %title(sprintf('Smoothed estimate of output %d',i))
    if i == N,
        legend([h1,h2,h3],'Prediction','Train data','Test data')
    end
    hold off;
    xlim([t_min t_max])
end

%% Plotting of latent forces
nmodels = length(model.Hi);
MS2fi = cell(1,nmodels);

PS2fi = cell(1,nmodels);

for i1 = 1:nmodels
    MS2fi{i1} = model.Hi{i1}*MS2;
    
    PS2fi{i1} = zeros(1,steps*isteps);
    
    for i = 1:size(PS2,3)
        PS2fi{i1}(i) = model.Hi{i1}*PS2(:,:,i)*model.Hi{i1}';
    end
end

for i = 1:nmodels
    figure(N+i); clf;
    fill([xx2' fliplr(xx2')], [(MS2fi{i}+1.96*sqrt(PS2fi{i})) ...
        fliplr((MS2fi{i}-1.96*sqrt(PS2fi{i})))], color1, 'edgecolor',color1);
    hold on
    
    h1=plot(xx2,MS2fi{i},'--k','LineWidth',1);

    h2 = plot(xx(mind==1),Y(N+i,:),'.k','LineWidth',1);
    
    flag = isnan(Y(N+i,:));
    if any(flag),
        h3 = plot(xx(flag),uq(i,flag),'.r','LineWidth',1);
    end
%     if i == nmodels
%         legend([h1,h2],'MAP parameters','True parameters')
%     end
    hold off;
    xlim([t_min t_max])
    %ylim(yl);
end

% Estimation of impulse response
Nd = length(tgrid);
hres = zeros(N,Nd);
htrue = hres;
Gamma = theta_opt(1:N);
C = reshape(theta_opt(N+1:N+D*N),N,D);
for d=1:N,
    gamma = Gamma(d);
    c = C(d,:)';
    hres(d,:) = sum(repmat(c, 1,length(tgrid)).*EvalLag(tgrid, D, gamma), 1);
    
    %b0 = params(3);
    %b1 = params(1);
    %w = .5*sqrt(b1^2 - 4*b0);
    %htrue = 1/w*exp(-.5*b1*tgrid).*sinh(w*tgrid);
    sys{d} = tf(1, params(d,:));
    htrue = impulse(sys{d},tgrid);
    figure;
    h1 = plot(tgrid,htrue,'--k','LineWidth',1);
    hold on
    h2 = plot(tgrid,hres(d,:),'.k','LineWidth',1);
end
legend([h1,h2],'MAP IRF','True IRF')