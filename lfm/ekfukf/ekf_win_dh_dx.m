% Jacobian of the measurement model function in the wiener demo
% using sine function
%
% Copyright (C) 2016 Cristian Guarnizo
%
% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.

function dY = ekf_win_dh_dx(x,param)
% param is H
mu = param*x;
dg = diag(cos(mu));

dY = dg*param;