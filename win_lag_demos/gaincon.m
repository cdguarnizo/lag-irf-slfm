function [c,ceq] = gaincon(x)
c = [];
w = (-1).^(0:9);
ceq = 1. - sqrt(2/exp(x(1)))*(sum(x(2:11).*w));