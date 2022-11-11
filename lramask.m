function [Xout]=lra(X,r)
    [U,S,V]=svd(X,'econ');
    for i=1:size(S,1)
        if r(i) == 0
            S(i,i)=0;
        end
    end
    Xout=U*S*V';
end