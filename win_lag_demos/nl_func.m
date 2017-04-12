function evalfun = nl_func(x,param,cas)
% This code is neccessary because of the update step in E_win_sde

iout = 1:param.nout;
iin = [];
if isfield(param,'incInput') && param.incInput,
    iin = param.nout + 1:param.nout + param.nlf;
end

% Consider missing data
if ~isfield(param,'flag') || isempty(param.flag),
    param.flag = true(size(param.H,1),1);
end

Ho = param.H(iout,:);
Hi = param.H(iin,:);
switch cas,
    case 1, %Evaluation of the nonlinear function
        M = param.H*x;
        evalfun = [param.g_func(M(iout,:),param); M(iin)];
    case 2, %Evaluation of gradient wrt x
        M = param.H*x;
        Ho = bsxfun(@times,param.dg_func(M(iout,:),param),Ho);
        evalfun = [Ho; Hi];
    case 3, %Evaluation of gradient wrt H*x
        evalfun = [param.g_u(x(iout),param); ones(param.nlf,1)];
    case 4, %second derivative wrt H*x
        evalfun = [param.g_u2(x(iout),param); zeros(param.nlf,1)];
end
evalfun = evalfun(param.flag,:);