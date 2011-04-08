function out = zorder2d(in)
% USAGE
%   out = zorder2d(in)
%
% NOTES
%
% - input must be square w. power-of-2 dimensions
% - self inverse apparently
% - recursive divide and conquer
%
% EXAMPLE
%
% >> I = reshape(1:16,4,[]);
% >> O = zorder2d(I);
% >> O
%
% O =
%
%       1     5     2     6     9    13    10    14     3     7     4     8    11    15    12    16
%
% >> O = zorder2d(I);
% >> zorder2d(reshape(O,4,[]))
%
%  ans =
%
%       1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16
%

%%
sz = size(in);
if any(sz>1)
  hsz = sz./2;
  ix0 = 1:hsz(1);
  iy0 = 1:hsz(2);
  ix1 = (hsz(1)+1):sz(1);
  iy1 = (hsz(2)+1):sz(2);
  
  a = zorder2d(in(ix0,iy0));
  b = zorder2d(in(ix1,iy0));
  c = zorder2d(in(ix0,iy1));
  d = zorder2d(in(ix1,iy1));
  
  out = [a b c d];
else
  out = in;
end