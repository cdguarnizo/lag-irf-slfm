% LAGUERRE_MODEL Model structure for latent force models
%
% Syntax:
%   model = laguerre_model(theta,param,calc_grad)
% 
% In:
%        theta - Parameters (frequency f and magnitude q of process noise) 
%        param - Number of harmonic components
%    calc_grad - Should the gradients be calculated
%     
% Out:
%        model - Structure containing the LTI model parameters and their
%                derivatives wrt given parameter vector theta               
% 
% Description:
%
%   Model structure for Laguerre approx. of latent force model (LFM) is 
%   described by the equations for outputs x_i(t), i=1,...,N:
% 
%   A_i d2x_i(t)/dt^2 + C_i dx_i(t)/dt + K_i x(t) = sum_{r=1}^R S_{i,r} u_r(t)
% 
%   where the latent forces u_r(t) have LTI SDE priors (augmented as a part
%   of the joint state-space model).
%
% Copyright (C) 2011-2012 Jouni Hartikainen
%
% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.
function model = laguerre_model_np(theta,param,calc_grad)

    model = struct;
    G  = [];
    
    D = param.D;                        % Number of basis for Laguerre
    N = param.N;                        % Number of outputs
    R = param.R;                        % Number of latent forces
    model.L = param.L;
    model.Nb = param.Nb;
    model_funcs  = param.model_funcs;   % Model structure for latent forces
    model_params = param.model_params;  % Fixed parameters of latent forces
    P0x = param.P0x;
    
    indp = 1:N;
    C = theta(indp);                    % Gamma of Laguerre functions
    indp = indp(end)+1:indp(end)+D*N;
    K = reshape(theta(indp),N,D);       % Laguerre coefficients
    indp = indp(end)+1:indp(end)+R*N;
    S = reshape(theta(indp),N,R);       % Sensitivities
    indp = indp(end)+1:indp(end)+model.Nb;
    model.w = reshape(theta(indp),model.Nb,1);
    indp = indp(end)+1;
    indp_lf = indp;
    
    S = ones(N,R);
    
    models = cell(1,length(model_funcs));
    nmodels = length(models);
    n = D*N;
    for i = 1:nmodels      
        models{i} = feval(model_funcs{i},theta(indp:end),model_params{i},1);
        indp = indp + size(models{i}.DF,3);        
        n = n + size(models{i}.F,1);
    end
    
    F = zeros(n,n);
    H = zeros(N,n);

    Qc = zeros(n,n);
    M0 = zeros(n,1);
    P0 = zeros(n,n);
    P0 = P0x;
       
    if calc_grad == 1
        DF = zeros(n,n,length(theta));
        DG = zeros(n,N,length(theta));
        DQc = zeros(n,n,length(theta));
        DP0 = zeros(n,n,length(theta));
        DM0 = zeros(n,length(theta));
        DH = zeros(N,n,length(theta));
        Dw = zeros(model.Nb,length(theta));
        Hgps = cell(1,R);   
    end
    
    % Model of outputs    
    ind = 0;
    for i = 1:N
        ind = ind(end)+1:ind(end)+D;
        % Building F
        F(ind,ind) = -(2*C(i))*(tril(ones(D,D),-1)) - C(i)*(diag(ones(1,D)));
        % Building H
        H(i,ind) = K(i,:);
    end
    
    indlf = D*N;
    
    % Indexes to needed outputs 
    indx = 1:D:D*N;
    
    Hi = cell(1,nmodels);
    
    indp = indp_lf;
    % Model for latent forces
    for r = 1:R
        gp_model = models{r};

        % Indexes to latent force components
        indlf = indlf(end)+1:indlf(end)+size(gp_model.F,1);        

        Hi{r} = zeros(1,n);
        Hi{r}(indlf) = gp_model.H;

        F(indlf,indlf)        = gp_model.F;
        Qc(indlf,indlf)       = gp_model.Qc;        
        M0(indlf)             = gp_model.M0; 
        P0(indlf,indlf)       = gp_model.P0;
        indpp = indp:indp+size(gp_model.DF,3)-1;

        if calc_grad == 1
            DF(indlf,indlf,indpp)  = gp_model.DF(:,:,1:size(gp_model.DF,3));
            DQc(indlf,indlf,indpp) = gp_model.DQc(:,:,1:size(gp_model.DF,3));
            DM0(indlf,indpp)       = gp_model.DM0(:,1:size(gp_model.DF,3));
            DP0(indlf,indlf,indpp) = gp_model.DP0(:,:,1:size(gp_model.DF,3));
            Hgps{r} = gp_model.H;
        end
        
        for i = 1:N
            F(indx(i):indx(i)+D-1,indlf(1)) = sqrt(2*C(i))*S(i,r)*ones(D,1);
        end
        indp = indp + size(gp_model.DF,3);
    end
    
    if param.incInput,
        H = [H; cell2mat(Hi')];
    end
    
    Qc = Qc + 10^-8*eye(n);
        
    % Store the parameters to model structure
    model.F  = F;
    model.G  = G;
    model.H  = H; 
    model.Qc = Qc;
    model.M0 = M0;
    model.P0 = P0;
    model.Hi = Hi;
    model.models = models;

    % Gradient of F wrt theta
    if calc_grad == 1
        
        % wrt C
        indp = 1:N;
        ind = 0;
        for i = 1:N
            ind = ind(end)+1:ind(end)+D;
            DF(ind,ind,indp(i)) = -2*(tril(ones(D,D),-1)) - diag(ones(1,D));
            indlf = N*D+1;
            for r = 1:R,
                DF(ind, indlf, indp(i)) = .5*sqrt(2/C(i))*S(i,r)*ones(D,1);
                indlf = indlf + size(Hgps{r},2);
            end
        end
        
        % wrt K
        indp = indp(end)+1:indp(end)+N*D;
        indp = reshape(indp,N,D);
        for i = 1:N,
            for d = 1:D,
                DH(i,(i-1)*D+d,indp(i,d)) = 1;
            end
        end       
        
        % wrt S
        indp = indp(end)+1:indp(end)+R*N;
        indp = reshape(indp,N,R);
        
        indlf = N*D+1;
        for r = 1:R
            for i = 1:N
                DF(indx(i):indx(i)+D-1,indlf,indp(i,r)) = sqrt(2*C(i))*ones(D,1);                
            end
            indlf = indlf+size(Hgps{r},2); 
        end        
        
        
        % wrt w
        indp = indp(end)+1:indp(end)+model.Nb;
        for i = 1:model.Nb,
            Dw(i,indp(i)) = 1;
        end
        
        % Gradient of Qc wrt theta (is zero)
        
        % Gradient of P0 wrt theta (is zero)
        
        % Store gradients
        model.DF  = DF;
        model.DG  = DG;
        model.DQc = DQc;
        model.DM0 = DM0;
        model.DP0 = DP0;
        model.DH  = DH;
        model.Dw  = Dw;
    end
end

