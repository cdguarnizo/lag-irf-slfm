% DE_LTI_SDE Gradient of the marginal likelihood of LTI SDEs
%
% Syntax:
%   DE = DE_lti_sde(theta,param)
%
% In:
%    theta - Parameters as 1xd or dx1 vector
%    param - Model parameter structure (see below for details) 
%
% Out:
%   DE - Derivative as 1xd vector
% 
% Description: 
%  Calculates the gradient of negative log marginal likelihood of LTI SDEs
%  of the form
%  
%   dx(t)/dt = F x(t) + G u(t) + L w(t),
%        y_k = H_k x(t_k) + r_k, r_k ~ N(0,R_k),
% 
%  where F,L,R_k and spectral density Qc of w(t) can depend parameter
%  vector theta. It is assumed that the user provides a function, which
%  calculates the gradients of these quantitites wrt theta (model_func 
%  field in the param structure). The constant input u(t) is assumed to
%  be known, and given in the param structure. Note: currently the gradient
%  for non-zero G can be calculated only if F is invertible!
%
%  See the following paper for more details on how to calculate the
%  derivatives:
% 
%  I. S. Mbalawata, S. Särkkä, and H. Haario (2012). Parameter Estimation
%  in Stochastic Differential Equations with Markov Chain Monte Carlo and
%  Non-Linear Kalman Filtering. Computational Statistics.
% 
% Copyright (C) 2012 Jouni Hartikainen, Simo Särkkä
%
% This software is distributed under the GNU General Public 
% Licence (version 3 or later); please refer to the file 
% Licence.txt, included with the software, for details.

function DE = DE_lti_sde(theta,param)

    % Transpose theta if given as column vector
    if size(theta,1) > size(theta,2)
        theta = theta';
    end

    Y           = param.Y;           % Measurements
    mind        = param.mind;        % Measurement index vector
    model_func  = param.model_func;  % Model function
    model_param = param.model_param; % Model function parameters
    R           = param.R;           % Measurement noise covariance (matrix or function returning matrix)
    H           = param.H;           % Measurement matrix (matrix or function returning matrix)
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
    
    % Add prior contribution and its gradient
    E_prior  = 0;
    DE_prior = zeros(1,length(theta)); 
    if isfield(param,'e_prior')
        for i = 1:length(theta)
            E_prior     = E_prior - real(feval(param.e_prior{i}.fh.lp,theta(i),param.e_prior{i}));
            DE_prior(i) = - real(feval(param.e_prior{i}.fh.lpg,theta(i),param.e_prior{i}));
        end
    end
    
    % Evaluate the parameters
    %
    if ~isnumeric(H)
        H = H(theta);
    end
    
    if isempty(mind)
        mind = ones(1,size(Y,2));
    end
    steps = length(mind);
    
    model = feval(model_func,theta,model_param,1);
    F   = model.F;
    G   = model.G;
    Qc  = model.Qc;
    DF  = model.DF;
    DG  = model.DG;
    DQc = model.DQc; 
    H   = model.H;
    % dH(theta)
    if isfield(model,'DH')    
        dH = model.DH;
    else
        dH = [];
    end
    
    if isempty(M0)
        M0  = model.M0;
        DM0 = model.DM0;
    else
        DM0 = zeros(size(F,1),length(theta));
    end
    if isempty(P0)
        P0  = model.P0;
        DP0 = model.DP0;
    else
        DP0 = zeros(size(F,1),size(F,1),length(theta));        
    end
    
    % Measurement covariance and its derivative
    if isempty(n_param) || isnumeric(R)
        DR = zeros(size(R,1),size(R,2),length(theta));
    else
        [R DR] = feval(R,theta,n_param);
    end
    
    % Add zero components for theta derivates not used by the dynamic model
    nzeros = length(theta)-size(DF,3);    
    for i = 1:nzeros
        DF(:,:,size(DF,3)+i)   = zeros(size(DF,1),size(DF,2));
        DQc(:,:,size(DQc,3)+i) = zeros(size(DQc,1),size(DQc,2));
        DG(:,:,size(DG,3)+i)   = zeros(size(DG,1),size(DG,2));
        DM0(:,size(DM0,2)+i)   = zeros(size(DM0,1),1);
        DP0(:,:,size(DP0,3)+i) = zeros(size(DP0,1),size(DP0,2));
    end

    dim = size(F,1);
    DE = DE_prior;

    Fmn = zeros(2*dim,2*dim,length(theta));
    Fcd = zeros(4*dim,4*dim,length(theta));
    
    dM  = zeros(dim,length(theta));
    dMp = zeros(dim,length(theta));
    dP  = zeros(dim,dim,length(theta));
    
    % Discretized model of M and P
    [A,Q] = lti_disc(F,[],Qc,dt);
     
    % Discretized input effect    
    if ~isempty(G) && ~isempty(U)
        Go = G;
        G = F\(A-eye(size(A)))*G;
        GG = zeros(2*size(G,1),size(G,2),length(theta));
    end
    
    % Parameters for the discretized model of dM and dP
    for k=1:length(theta)
        n0 = DM0(:,k);
        B0 = DP0(:,:,k);
        
        FF = [F zeros(dim,dim);
            DF(:,:,k) F];
        
        Fmn(:,:,k) = expm(FF*dt);
        
        FFF = [F Qc zeros(dim,dim) zeros(dim,dim);
               zeros(dim,dim) -F' zeros(dim,dim) zeros(dim,dim);
               DF(:,:,k) DQc(:,:,k) F Qc;
               zeros(dim,dim) -DF(:,:,k)' zeros(dim,dim) -F'];
        
        Fcd(:,:,k) = expm(FFF*dt);
        dM(:,k)   = n0;
        dP(:,:,k) = B0;
        
        if ~isempty(G) && ~isempty(U)
            GG(:,:,k) = FF\(Fmn(:,:,k)-eye(size(Fmn,1)))*[Go;DG(:,:,k)];
        end
    end
    
    M = M0;
    P = P0;
    
    % Measurement counter
    mc = 1;
    
    % Logical indexing (speeds up a bit)
    ind31 = zeros(3*dim,1);
    ind31(1:dim) = 1;
    ind31 = logical(ind31);
    
    ind42 = zeros(4*dim,1);
    ind42(dim+1:end) = 1;
    ind42 = logical(ind42);
    
    ind413 = zeros(4*dim,1);
    ind413(1:3*dim) = 1;
    ind413 = logical(ind413);
    
    ind11 = true(dim,1);
    
    ind22 = zeros(2*dim,1);
    ind22(dim+1:end) = 1;
    ind22 = logical(ind22);
    
    for i=1:steps
        % Prediction of M and P
        M_prev = M;
        P_prev = P;
        
        if i > start_ind
            if isempty(G) || isempty(U)
                [M,P] = kf_predict(M,P,A,Q);
            else
                [M,P] = kf_predict(M,P,A,Q,G,U(:,i));
            end
            
            for k = 1:length(theta);
                
                % Prediction of dM and dP
                if isempty(DG) || isempty(U)
                    dM(:,k) = Fmn(ind22,:,k)*[M_prev;dM(:,k)];
                else
                    dM(:,k) = Fmn(ind22,:,k)*[M_prev;dM(:,k)] + GG(ind22,:,k)*U(:,i);
                end
                
                CDXY = Fcd(ind42,ind413,k)*[P_prev;eye(dim);dP(ind11,ind11,k)];
                D    = CDXY(ind31,:);
                XX   = CDXY(dim+1:2*dim,:);
                YY   = CDXY(2*dim+1:end,:);
                dP(:,:,k) = (XX-P*YY)/D;
                
                % These are needed in derivative update / energy
                dMp(:,k) = dM(:,k);
            end
        end
        Pp = P;

        
        % Update
        if mind(i) == 1,
            Yt = Y(:,mc);
            %Missing data setup
            flag = ~isnan(Yt); % Missing data
            Yt = Yt(flag);
            Ht = H(flag,:);
            dHt = dH(flag,:,:);
            Rt = R(flag,flag);
            DRt = DR(flag,flag,:);
            
            IM = Ht*M;
            S = (Rt + Ht*P*Ht');
            HtiS = Ht'/S;
            K = P*HtiS;
            mu = Yt-IM;
            Mp = M;
            M = M + K * mu;
            
            SKt = S*K';
            P = P - K*SKt;
            
            KH = K*Ht;
            iS = inv(S);
            YmIMiS = mu'/S;
            YmIMiSH = YmIMiS*Ht;
            
            for k = 1:length(theta)
                
                if isempty(dHt),
                    dS = Ht*dP(:,:,k)*Ht'+ DRt(:,:,k);
                    dK = dP(:,:,k)*HtiS-Pp*HtiS*dS/S;
                    dM(:,k) = dM(:,k) + dK*mu - KH*dM(:,k);
                    
                    dKSKt = dK*SKt;
                    dP(:,:,k) = dP(:,:,k) - dKSKt - K*dS*K' - dKSKt';

                    YmIMiSHdMp = YmIMiSH*dMp(:,k);
                    YmIMiSdHMp = YmIMiS*dH(:,:,k)*Mp;
                    DE(k) = DE(k) + 0.5*sum(sum(iS.*dS)) ...
                    -0.5*YmIMiSHdMp' ...
                    -0.5*YmIMiS*dS*YmIMiS' ...
                    -0.5*YmIMiSHdMp;
                else
                    dHPHt = dHt(:,:,k)*Pp*Ht';
                    dS = Ht*dP(:,:,k)*Ht'+ DRt(:,:,k) + dHPHt + dHPHt';
                    dK = dP(:,:,k)*HtiS + (- Pp*HtiS*dS + Pp*dHt(:,:,k)')/S;
                    dM(:,k) = dM(:,k) + dK*mu - KH*dM(:,k) - K*dHt(:,:,k)*Mp;
                    
                    dKSKt = dK*SKt;
                    dP(:,:,k) = dP(:,:,k) - dKSKt - K*dS*K' - dKSKt';

                    YmIMiSHdMp = YmIMiSH*dMp(:,k);
                    YmIMiSdHMp = YmIMiS*dHt(:,:,k)*Mp;
                    DE(k) = DE(k) + 0.5*sum(sum(iS.*dS)) ...
                    -0.5*YmIMiSHdMp' ...
                    -0.5*YmIMiS*dS*YmIMiS' ...
                    -0.5*YmIMiSHdMp ...
                    -0.5*YmIMiSdHMp -0.5*YmIMiSdHMp';
                end
            end
            % Increment measurement counter
            mc = mc + 1;
        end
    end
    
    % Multiply the gradient with theta values due to log-transform
    theta = reshape(theta,size(DE));
    DE(ltr_ind) = DE(ltr_ind).*theta(ltr_ind);