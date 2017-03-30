function L = EvalLag(t,N,p)
[Ap,L0]=lagc(p,N);
L = zeros(N,length(t));
%solution of the differential equation
for i=1:length(t);
    L(:,i)=expm(Ap*t(i))*L0;
end


    function [Ap,L0]=lagc(p,N)
        %Generating system matrix Ap
        Ap=-p*eye(N,N);
        for ii=1:N
            for jj=1:N
                if jj<ii,
                    Ap(ii,jj)=-2*p;
                end
            end
        end
        L0=sqrt(2*p)*ones(N,1);
    end

end