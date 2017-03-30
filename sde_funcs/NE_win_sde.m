% E_Win_SDE Negative log marginal likelihood of LTI SDEs
% 
% Syntax:
%  E = E_Win_sde(theta,param)
%
% In:
%   theta - Parameters as dx1 vector
%   param - Model parameter structure (see below for details)
% 
% Out:
%   E - Negative log likelihood
%
% Description: 
%  Calculates the negative log marginal likelihood of Wiener SDEs
%  of the form
%  
%   dx(t)/dt = F x(t) + G u(t) + L w(t),
%        y_k = H_k x(t_k) + r_k, r_k ~ N(0,R_k),
%
%  See the following paper for more details:
% 
%  I. S. Mbalawata, S. Särkkä, and H. Haario (2012). Parameter Estimation
%  in Stochastic Differential Equations with Markov Chain Monte Carlo and
%  Non-Linear Kalman Filtering. Computational Statistics.
%
% Copyright (C) 2016 Cristian Guarnizo
% Based on codes from S. Särkkä and J. Hartikainen
%
% This software is distributed under the GNU General Public 
% Licence (version 3 or later); please refer to the file 
% Licence.txt, included with the software, for details.
function E = NE_win_sde(theta,param)
    % Transpose theta if given as column vector
    if size(theta,1) > size(theta,2)
        theta = theta';
    end

    Y           = param.Y;           % Measurements
    mind        = param.mind;        % Measurement index vector
    model_func  = param.model_func;  % Model function
    model_param = param.model_param; % Model function parameters
    R           = param.R;           % Measurement noise covariance (matrix or function returning matrix)
%    H           = param.H;           % Measurement matrix (matrix or function returning matrix)
    dt          = param.dt;          % Time step between measurements

    % Parameters of R if it's a function
    if isfield(param,'n_param')
        n_param = param.n_param;
    else
        n_param = [];
    end
    
    % Prior mean
    if isfield(param,'M0')    
        M0 = param.M0;
    else
        M0 = [];
    end
    
    % Prior covariance
    if isfield(param,'P0')
        P0 = param.P0;
    else
        P0 = [];
    end
    
    % Control input
    if isfield(param,'U')    
        U = param.U;
    else
        U = [];
    end
    
    % Indicator vector for log-transformed parameters    
    if isfield(param,'ltr_ind')
        ltr_ind = param.ltr_ind;
    else
        ltr_ind = 1:length(theta);
    end
    theta(ltr_ind) = exp(theta(ltr_ind));        
    
    if isfield(param,'start_ind')
        start_ind = param.start_ind;
    else
        start_ind = 0;
    end
    
    % Add prior contribution
    E_prior = 0;
    if isfield(param,'e_prior')
        for i = 1:length(theta)
            E_prior = E_prior - real(feval(param.e_prior{i}.fh.lp,theta(i),param.e_prior{i}));
        end
    end
    
    %
    % Evaluate the parameters
    %
%     if ~isnumeric(H)
%         H = H(theta);
%     end
    if ~isnumeric(R)
        R = R(theta,n_param);
    end
    if isempty(mind)
        mind = ones(1,size(Y,2));
    end
    steps = size(mind,2);
    
    % LTI SDE model parameters
    model = feval(model_func,theta,model_param,0);
    F  = model.F;
    G  = model.G;
    Qc = model.Qc;
    
    if isfield(model,'w'),
        par.w = model.w;
        par.H = model.H;
    else
        par.H = model.H;
    end
    
    if isempty(M0)
        M0 = model.M0;
    end
    if isempty(P0)
        P0 = model.P0;
    end

    % Discretized model
    [A,Q] = lti_disc(F,[],Qc,dt);
    
    % Discretized input effect
    if ~isempty(G)
        G = F\(A-eye(size(A)))*G;
    end
    
    % Measurement counter
    mc = 1;
        
    % Filtering
    E = E_prior;
    M = M0;
    P = P0;
    par.H = model.H;
    par.w = model.w;
    for k = 1:steps
        if k > start_ind
            if isempty(G) || isempty(U)
                [M,P] = kf_predict(M,P,A,Q);
            else
                [M,P] = kf_predict(M,P,A,Q,G,U(:,k));
            end
        end
        if mind(k) == 1,
            Yt = Y(:,mc);
            flag = ~isnan(Yt);
            Yt = Yt(flag);
            Rt = R(flag,flag);
            par.flag = flag;
            
            [M,P,~,IM,S]= ekf_update1(M,P,Y(:,mc),param.Gf,R*eye(1),param.gf,[],par);
            E = E + 0.5*log(det(2*pi*S))+0.5*(Y(:,mc)-IM)'/S*(Y(:,mc)-IM);
            mc = mc + 1;
        end
    end
   
