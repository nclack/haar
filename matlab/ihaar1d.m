function out = haar1dinv(in)

NN = numel(in);
out = in;
N=2;
while(N<=NN)
  [t,out(1:N)] = kernel(out(1:N));
  N=t;
end

end

%%
% NOTE:
%
% See haar1d for comments on forward vs inverse kernel

function [N,out] = kernel(in)
NN = numel(in);
N=NN/2;
t=zeros(size(in));
t(1:2:NN) = in(1:N)+in((N+1):NN);
t(2:2:NN) = in(1:N)-in((N+1):NN);
t(1:NN  ) = t(1:NN).*0.70710678118654752440;
N   = 2*NN;
out = t;
end