function out = ihaar(in)

sz  = min( [repmat(2,[1 ndims(in)]); size(in)] );
szt = sz;
out = in;
p   = circshift((1:ndims(in))',-1);
ds  = domains(size(in));
for ids=1:size(ds,1)
  sz = ds(ids,:);
  for d = 1:numel(sz)
    if(sz(1)>1)
      S = subs(sz);
      [szt(d),t] = kernel(subsref(out,S));
      out = subsasgn(out,S,t);
    end
    out = permute(out,p);
    sz  = sz(p);
  end
end
end

%%
function s = domains(sz)
o = ones(size(sz));
n = max(log2(sz));
s = zeros(n,numel(sz));
for i=1:n
  s(i,:) = sz;
  sz = max([sz/2.0;o]);
end
s = flipud(s);
end

%%
function [N,out] = kernel(in)
% adapted for multidimensional input. reduction on rows
NN = size(in,1);
N=NN/2;
t=zeros(size(in));
t(1:2:NN,:) = in(1:N,:)+in((N+1):NN,:);
t(2:2:NN,:) = in(1:N,:)-in((N+1):NN,:);
t(  1:NN,:) = t(1:NN,:).*0.70710678118654752440;
N   = 2*NN;
out = t;
end

%%
function S = subs(sz)
c = cell([1 numel(sz)]);
for i=1:numel(sz)
  c{i} = 1:sz(i);
end
S = substruct('()',c);
end