function evalfun = nl_func_np(x,param)

switch cas
    case 1, %Evaluation of the nonlinear function
        evalfun = @(x, param) (L^(-1/2)*sin( pi*in*(param.H*x + L)/(2*L) )) * param.w(:);
    case 2, %Evaluation of gradient wrt x
        elvafun = @(x, param) L^(-1/2)* bsxfun(@times,( cos( pi*in*(param.H*x + L)/(2*L) ).*(pi*in/(2*L)) )*param.w(:), param.H);
    case 3, %Evaluation of gradient wrt H*x
        evalfun = @(x, param)  L^(-1/2)*( cos( pi*in*(x + L)/(2*L) ).*(pi*in/(2*L)) )*param.w(:);
    case 4, %Hessian wrt H*x
        evalfun = @(x, param)  L^(-1/2)*( -sin( pi*in*(x + L)/(2*L) ).*(pi*in/(2*L)).^2 )*param.w(:);
end