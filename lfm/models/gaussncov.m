% GAUSSNCOV Diagonal multidimensional Gaussian noise model
%
% Syntax:
%   [R,dR] = gaussncov(theta,param)
% 
% In:
%        theta - Parameters (frequency f and magnitude q of process noise) 
%        param - Number of harmonic components
%    calc_grad - Should the gradients be calculated
%     
% Out:
%        R - Noise covariance matrix
%       dR - Derivatives of R wrt theta
% 
% Description:
%
%
% Copyright (C) 2011-2012 Jouni Hartikainen
%
% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.

% Diagonal (multidimensional) Gaussian noise
function [R,dR] = gaussncov(theta,param)

    % Number of observations
    n   = param{1};
    
    % Vector telling which theta value to use with each output component
    ind = param{2};
    
    % Noise covariance
    R = zeros(n,n);
    for i = 1:length(ind)
        R(i,i) = theta(ind(i)).^2;
    end
        
    % Gradient of R 
    if nargout > 1
        dR = zeros(n,n,length(theta));
        for i = 1:length(ind)
            dR(i,i,ind(i)) = 2*theta(ind(i));
        end
    end

end

