function out = haar1d(in)

NN = numel(in);
out = in;
N=NN;
while(N>1)
  [N,out(1:NN)] = kernel(out(1:NN));
  NN=N;
end

end

%%
% NOTE:
%
% The forward and inverse kernels can be self-inverse.  There
% are two differences in the implementation here.
% 1. the input and output strides used for the vectorized
%    transform.
% 2. the generation of the next 'N'

function [N,out] = kernel(in)
NN = numel(in);
N=NN/2;
t=zeros(size(in));
t(    1:N  ) = in(1:2:NN)+in(2:2:NN);
t((N+1):NN ) = in(1:2:NN)-in(2:2:NN);
t(    1:NN ) = t(1:NN).*0.70710678118654752440;
out = t;
end