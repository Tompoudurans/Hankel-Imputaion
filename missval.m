N=50;


sigma=0.1;
N_sim=1;
L=10;% L =rows and k =colums n is size m is missing end
k=5;

    for n=1:N
        s(n)=cos(2*pi*n/10); % cacd se 1
        %s(n)=n/10
        %s(n)=cos(2*pi*n/10)*exp(0.02*n); % case 2
   end

m = 15;
r=sigma*randn(N,1); 
Y=s+r';
X=hmat(Y(1:end-m),L);


wF=froweights(L,size(X,2));
wUnit=ones(N-m,1);

clear wExp;
for i=1:N-m
wExp(i)=1.03^i;
end
wExp=wExp';

tauF=sqrt(wvnorm(hankvec_avg(lra(X,k))-Y(1:end-m)',wF))
tauUnit=sqrt(wvnorm(hankvec_avg(lra(X,k))-Y(1:end-m)',wUnit))
tauExp=sqrt(wvnorm(hankvec_avg(lra(X,k))-Y(1:end-m)',wExp))


Ya=mcwf(Y(1:end-m)',L,m,wF,tauF);
Ya_unit=mcwf(Y(1:end-m)',L,m,wUnit,tauUnit);
Ya_exp=mcwf(Y(1:end-m)',L,m,wExp,tauExp);
