function evalfun = nl_func(x,param,cas)
% This code is neccessry because of the update step in E_win_sde

if isfield(param,'incInput') && param.incInput,
    iout = 1:param.nout;
    iin = param.nout + 1:param.nout + param.nlf;
end

if ~isfield(param,'flag') || isempty(param.flag),
    param.flag = true(size(param.H,1),1);
end

Ho = param.H(iout,:);
Hi = param.H(iin,:);
switch cas
    case 1, %Evaluation of the nonlinear function
        M = param.H*x;
        evalfun = [param.g_func(x,Ho); M(iin)];
    case 2, %Evaluation of gradient wrt x
        evalfun = [param.dg_dx_func(x,Ho); Hi];
    case 3, %Evaluation of gradient wrt H*x
        evalfun = [param.g_u(x(iout)); ones(param.nlf,1)];
    case 4, %secodn derivative wrt H*x
        evalfun = [param.g_u2(x(iout)); zeros(param.nlf,1)];
end
evalfun = evalfun(param.flag,:);