#include <string.h>

#define SQRT2    1.41421356237309504880
#define INVSQRT2 0.70710678118654752440

/** UTIL **********************************************************************
*                                                                             *
* LOG2                                                                        *
* ----                                                                        *
* Taken from http://graphics.stanford.edu/~seander/bithacks.html#IntegerLog   *
* O(lg(nbits))                                                                *
* Divide and conquer with table lookup over last 8 bits.                      *
*                                                                             *
******************************************************************************/

static inline 
unsigned u32log2(uint32_t v)
{
  static const char LogTable256[256] = 
  {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
    LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
  };

  unsigned r;                  // r will be lg(v)
  register uint32_t t, tt;     // temporaries

  if (tt = v >> 16)
  {
    r = (t = tt >> 8) ? 24 + LogTable256[t] : 16 + LogTable256[tt];
  }
  else 
  {
    r = (t = v >> 8) ? 8 + LogTable256[t] : LogTable256[v];
  }
  return r;
}

static inline
unsigned u64log2(uint64_t v)
{
  return (tt=v>>32)?(32+u32log2((uint32_t)tt)):u32log2((uint32_t)v);
}

/******************************************************************************
*  VECTOR OPS *****************************************************************
******************************************************************************/

typedef T float;

// zs = xs + ys
inline 
void vadd(size_t N, T* zs, size_t stz, T* xs, size_t stx, T* ys, size_t sty)
{ size_t i;
  for(i=0;i<N;++i)
    zs[i*stz] = xs[i*stx] + ys[i*sty];
}

// zs = xs - ys
inline 
void vsub(size_t N, T* zs, size_t stz, T* xs, size_t stx, T* ys, size_t sty)
{ size_t i;
  for(i=0;i<N;++i)
    zs[i*stz] = xs[i*stx] - ys[i*sty];
}

// zs .*= a;
inline 
void vmul_scalar_ip(size_t N, T* zs, size_t stz, T a)
{ size_t i;
  for(i=0;i<N;++i) zs[i*stz] *= a;
}

/******************************************************************************
*  KERNEL  ********************************************************************
******************************************************************************/

// haar kernel: in(1:N) --> out(1:N)
// ---------------------------------
// 'out' and 'in' must not overlap
// 'st*' are the strides
//
// The matrix multiplication underlying this is self-inverse so 
// the kernels end up looking very similar!
static
void kernel(size_t N, T* out, size_t stout, T* in, size_t stin)
{ const size_t halfN = N/2;                                   // e.g. (matlab) - the precise indexing is determined by strides
  vadd(halfN,out            ,stout,in,2*stin,in+stin,2*stin); // out(    1:N  ,:) = in(1:2:NN,:)+in(2:2:NN,:);
  vsub(halfN,out+halfN*stout,stout,in,2*stin,in+stin,2*stin); // out((N+1):NN ,:) = in(1:2:NN,:)-in(2:2:NN,:);
  vmul_scalar_ip(N,out,INVSQRT2);
}

static
void ikernel(size_t N, T* out, size_t stout, T* in, size_t stin)
{ const size_t halfN = N/2;                                   // e.g. (matlab) - the precise indexing is determined by strides
  vadd(halfN,out      ,2*stout,in,stin,in+halfN*stin,stin);   // out(1:2:NN,:) = in(1:N,:)+in((N+1):NN,:);
  vsub(halfN,out+stout,2*stout,in,stin,in+halfN*stin,stin);   // out(2:2:NN,:) = in(1:N,:)-in((N+1):NN,:);
  vmul_scalar_ip(N,out,INVSQRT2);
}

/******************************************************************************
*  DOMAINS  *******************************************************************
*
*  Usage: 
*  Input - ndim,*dims
*  { DomainList dl;
*    memset(dl,0,sizeof(dl));
*    GetDomains(&dl,ndim,dims);
*    // Use result
*    DomainListClean(&dl);
*  }
******************************************************************************/

typedef struct tagDomainList
{ size_t *shapes;
  size_t sz[2];
  size_t cursor;
} DomainList;

static
void DomainListRealloc(DomainList *self, size_t ndim, size_t n)
{ if(self->shapes)
    self->shapes = realloc(self->shapes,sizeof(size_t)*ndim*n);
  else
    self->shapes = malloc(sizeof(size_t)*ndim*n);
  self->sz[1]=n;    //rows
  self->sz[0]=ndim; //cols
}

static
void DomainListClean(DomainList *self)
{ if(self)
  { if(self->shape) free(self->shape);
    self->shape=0;
  }
}

static
void GetDomains(DomainList* out, size_t ndim, size_t* dims)
{ unsigned n=0;
  size_t i,j;

  // compute number of iterations
  for(i=0;i<ndim;++i)
  { unsigned v = log2(dims[i]);
    n = max(n,v);
  }

  DomainListRealloc(out,ndim,n);

  // compute domains
  size_t *cursor = out->shapes;
  for(j=0;j<ndim;++j)
    *cursor++ = dims[i];
  for(i=1;i<n;++i)
    for(j=0;j<ndim;++j)
      *cursor++ = max(cursor[-ndim]>>1,1);
}

static inline
void DomainListResetCursor(DomainList *self)
{ self->cursor=0;
}

static inline
unsigned* NextDomain(DomainList* self)
{ size_t t = self->cursor;
  size_t ndim = self->sz[1];
  self->cursor = (t+1)%(ndim+1);   //inc - one extra for end of sequence
  if(t==ndim) 
    return 0;
  return self->shapes[t*self->sz[0]];
}

static inline
unsigned* PrevDomain(DomainList* self)
{ size_t t = self->cursor;
  size_t ndim = self->sz[1];
  self->cursor = (t-1)%(ndim+1);   //inc - one extra for end of sequence
  if(t==0) 
    return 0;
  return self->shapes[(t-1)*self->sz[0]];
}

/**  TRANSFORM  ***************************************************************
* 
* o General interface
* - Out-of-place
* - Usage:
*   Input - ndim, shape[ndim], out, ostrides[ndim], in, istrides[ndim]
*   { HaarWorkspace ws; //size is ~ sizeof(size_t) * log2(max(dims)) * ndims
*     haar(&ws,ndim,shape,ot,ostrides,in,istrides);
*     ihaar(&ws,ndim,shape,ot,ostrides,in,istrides); 
*     HaarWorkspaceClean(&ws); // when done
*   }
* - Notes:
*   By manipulating shape and setting out and in to the proper point, it's
*   possible to restrict the transform to subvolumes.
* [ ] TODO: In-place
*
* o Mylib Array interface
*   [ ] TODO
* o Matlab interface
*   [ ] TODO
******************************************************************************/

typedef struct tagHaarWorkspace
{
  DomainList domains;
} HaarWorkspace;

void HaarWorkspaceClean(HaarWorkspace* ws)
{ DomainListClean(&wd->domains);
}

void haar(HaarWorkspace* ws, size_t ndim, size_t* shape, T* out, size_t* ostrides, T* in, size_t* istrides)
{ DomainList *domains;
  size_t *domain,i;
  
  domains = &ws->domains;
  DomainListResetCursor(domains);
  GetDomains(domains,ndim,shape);
  while(domain=NextDomain(domains))
    for(i=0;i<ndim;++i)
      kernel(domain[i],out,ostrides[i],in,istrides[i]);
}

void ihaar(HaarWorkspace* ws, size_t ndim, size_t* shape, T* out, size_t* ostrides, T* in, size_t* istrides)
{ DomainList *domains;
  size_t *domain,i;
  
  domains = &ws->domains;
  DomainListResetCursor(domains);
  GetDomains(domains,ndim,shape);
  while(domain=PrevDomain(domains))
    for(i=0;i<ndim;++i)
      ikernel(domain[i],out,ostrides[i],in,istrides[i]);
}
