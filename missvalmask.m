clear

sigma=0.1;
N_sim=1;
s = importdata("121995_weekly.csv");
Y = s.data(:,1);
N = length(Y);
L=round(N/2);
TF = isnan(Y);
   for n=1:N
       if TF(n) == 1
           Y(n) = 0;
       end
   end
X=hmat(Y(1:end),L);
wF=froweights(L,size(X,2));
wUnit=ones(N,1);
c = lramask(X,TF);
tauF=sqrt(wvnorm(hankvec_avg(lramask(X,TF))-Y(1:end) ,wF));
Ya=mcw(Y(1:end),L,wF,tauF);

