% LFM2_MODEL Model structure for a 2nd order latent force model
%
% Syntax:
%   model = lfm2_model(theta,param,calc_grad)
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
%   Model structure for a 2nd order latent force model (LFM) described by
%   the equations for outputs x_i(t), i=1,...,N:
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
function model = lfm2_model(theta,param,calc_grad)

    model = struct;
    G  = [];
    
    N = param.N;                        % Number of outputs
    R = param.R;                        % Number of latent forces
    model_funcs  = param.model_funcs;   % Model structure for latent forces
    model_params = param.model_params;  % Fixed parameters of latent forces
    P0x = param.P0x;
    M0x = param.M0x;   
    
    A = ones(N,1);  

    indp = 1:N;
    C = theta(indp);
    indp = indp + N;  
    K = theta(indp);    
    indp = indp(end)+1:indp(end)+R*N;    
    S = reshape(theta(indp),N,R);
    indp = indp(end)+1;
    indp_lf = indp;

    models = cell(1,length(model_funcs));
    nmodels = length(models);
    n = 2*N;
    for i = 1:nmodels      
        models{i} = feval(model_funcs{i},theta(indp:end),model_params{i},1);
        indp = indp + size(models{i}.DF,3);        
        n = n + size(models{i}.F,1);
    end
    
    F = zeros(n,n);
    H = zeros(N,n);
    for i = 1:N
        H(i,(i-1)*2+1) = 1;
    end
    Qc = zeros(n,n);
    M0 = zeros(n,1);
    M0(1:2*N) = M0x;
    P0 = zeros(n,n);
    P0(1:2*N,1:2*N) = P0x;
       
    if calc_grad == 1
        DF = zeros(n,n,length(theta));
        DG = zeros(n,N,length(theta));
        DQc = zeros(n,n,length(theta));
        DP0 = zeros(n,n,length(theta));
        DM0 = zeros(n,length(theta));
        Hgps = cell(1,R);   
    end
    
    % Model of outputs    
    ind = 0;
    for i = 1:N
        ind = ind(end)+1:ind(end)+2;
        F(ind,ind) = [0 1;-K(i)./A(i) -C(i)./A(i)];
    end
    
    indlf = 2*N;
    
    % Indexes to needed outputs 
    indx = 2:2:2*N;
    
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
            F(indx(i),indlf) = S(i,r)./A(i)*gp_model.H;
        end
        indp = indp + size(gp_model.DF,3);
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
        %indp = indp + N;
        indp = 1:N;
        ind = 0;
        for i = 1:N
            ind = ind(end)+1:ind(end)+2;
            DF(ind,ind,indp(i)) = [0 0;0 -1./A(i)];
        end
        
        % wrt K
        indp = indp + N;
        ind = 0;
        for i = 1:N
            ind = ind(end)+1:ind(end)+2;
            DF(ind,ind,indp(i)) = [0 0;-1./A(i) 0];
        end
        
        % wrt S
        indp = indp(end)+1:indp(end)+R*N;
        indp = reshape(indp,N,R);
        
        indlf = 2*N;
        for r = 1:R
            %indlf = indlf(end)+1:indlf(end)+p(r)+1;
            indlf = indlf(end)+1:indlf(end)+size(models{r}.F,1);        
            for i = 1:N
                DF(indx(i),indlf,indp(i,r)) = 1./A(i)*Hgps{r};
            end
        end        
        
        % Gradient of Qc wrt theta (is zero)
        
        % Gradient of P0 wrt theta (is zero)
        
        % Store gradients
        model.DF  = DF;
        model.DG  = DG;
        model.DQc = DQc;
        model.DM0 = DM0;
        model.DP0 = DP0;
    end
end

