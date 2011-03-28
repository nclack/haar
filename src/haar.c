#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include <stdio.h>

#define SQRT2    1.41421356237309504880
#define INVSQRT2 0.70710678118654752440

typedef float T;

/** UTIL **********************************************************************
*                                                                             *
* LOG2                                                                        *
* ----                                                                        *
* Taken from http://graphics.stanford.edu/~seander/bithacks.html#IntegerLog   *
* O(lg(nbits))                                                                *
* Divide and conquer with table lookup over last 8 bits.                      *
*                                                                             *
* COPY                                                                        *
* ----                                                                        *
* Recursively copy linear-layout nd-array.                                    *
* Attempts to optimized for contiguous blocks.                                *
* IMPORTANT - this routine imposes an ordering for how the strides and shape. *
*             The convention is that the lowest index should correspond to the*
*             smallest 'natural' stride.  That is, an N-element 1D array      *
*             would have stride: {1, N}.                                      *
******************************************************************************/

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

  if ((tt = v >> 16))
  {
    r = (t = tt >> 8) ? 24 + LogTable256[t] : 16 + LogTable256[tt];
  }
  else 
  {
    r = (t = v >> 8) ? 8 + LogTable256[t] : LogTable256[v];
  }
  return r;
}

unsigned u64log2(uint64_t v)
{ register uint64_t tt;
  return (tt=v>>32)?(32+u32log2((uint32_t)tt)):u32log2((uint32_t)v);
}

static
void copy_recurse(int64_t m, int64_t idim,size_t *shape,T* out,size_t *ostrides,T *in,size_t *istrides)
{ size_t i,os,is;
  if(idim<0) { *out = *in; return; }
  if(idim<m) 
  { size_t nelem = shape[idim];
    while(idim--)
      nelem *= shape[idim]; // FIXME?? suspect that nelem == ostrides[m] bc of conditions on m
    memcpy(out,in,sizeof(T)*nelem);
    return;
  } 
  os = ostrides[idim];
  is = istrides[idim];
  for(i=0;i<shape[idim];++i)
    copy_recurse(m,idim-1,shape,out+os*i,ostrides,in+is*i,istrides);
}

void copy(size_t ndim,size_t *shape,T* out,size_t *ostrides,T *in,size_t *istrides)
{ 
  int64_t m;
  size_t *oshape,*ishape;
  // 0. Nothing to do?
  if(out==in) return;
  // 1. Find max index such that a block copy is ok
  //      ostrides[j]==istrides[j] and oshape[j]==ishape[j]==shape[j] for j<i
  //    where oshape and ishape are the shapes derived from the input
  //      strides.
  if(!(oshape=alloca(sizeof(size_t)*ndim*2))) goto MemoryError;
  ishape=oshape+ndim;
  // stride[i] = shape[i-1]*shape[i-2]*...*shape[0]
  // so shape[i]  = stride[i+1]/stride[i]
  for(m=0;m<ndim;++m)
  { ishape[m] = istrides[m+1]/istrides[m];
    oshape[m] = ostrides[m+1]/ostrides[m]; 
  }
  m=0;
  if(ostrides[0]==1 && istrides[0]==1)
    for(m=0;m<ndim 
        && ostrides[m]==istrides[m]
        &&   oshape[m]==ishape[m]
        &&    shape[m]==oshape[m]
        ;++m);

  // 2. Recursively copy on outer dimension (max st shape is not 1)
  //    use memcpy when dimension gets to dimension i.
  copy_recurse(m,ndim-1,shape,out,ostrides,in,istrides);
  return;
MemoryError:
  fprintf(stderr,"Could not allocate block of %zu elements on stack.",2*ndim);
  exit(1);
}

/******************************************************************************
*  ND ARRAY OPS ***************************************************************
******************************************************************************/
typedef void (   binary_vec_op_t)(size_t,T*,size_t,T*,size_t,T*,size_t);    // z[:] <- x[:] op y[:]
typedef void (scalar_ip_vec_op_t)(size_t,T*,size_t,T);                      // z[:] .op a

static
void binary_op_recurse(
    int64_t m, 
    int64_t idim, 
    size_t *shape,
    T* z,size_t *zst,
    T* x,size_t *xst, 
    T* y,size_t *yst, 
    binary_vec_op_t *f)
{ size_t i,ox,oy,oz;
  if(idim<m) 
  { 
    f(zst[m]/zst[0],z,zst[0],x,xst[0],y,yst[0]);
    return;
  } 
  ox = xst[idim];
  oy = yst[idim];
  oz = zst[idim];
  for(i=0;i<shape[idim];++i)
    binary_op_recurse(m,idim-1,shape,z+oz*i,zst,x+ox*i,xst,y+oy*i,yst,f);
}

static
void binary_op(
    size_t ndim,
    size_t *shape,
    T* z,size_t *zst,
    T* x,size_t *xst, 
    T* y,size_t *yst, 
    binary_vec_op_t *f)
{ 
  int64_t m;
  size_t *zshape,*xshape,*yshape;

  // 1. Find max index such that lower dimension indexes may be collapsed(vectorized)
  //      ostrides[j]==istrides[j] and oshape[j]==ishape[j]==shape[j] for j<i
  //    where oshape and ishape are the shapes derived from the input
  //      strides.
  if(!(zshape=alloca(sizeof(size_t)*ndim*3))) goto MemoryError;
  xshape=zshape+  ndim;
  yshape=zshape+2*ndim;
  for(m=0;m<ndim;++m)                                                           // compute native shapes from strides
  { zshape[m] = zst[m+1]/zst[m];
    xshape[m] = xst[m+1]/xst[m];
    yshape[m] = yst[m+1]/yst[m];
  }
  m=0;
  for(m=0;m<ndim
      && zst[m]==xst[m]
      && zst[m]==yst[m]
      && zshape[m]==xshape[m]
      && zshape[m]==yshape[m]      
      && zshape[m]== shape[m]
      ;++m);

  // 2. Recursively operate on outer dimension (max st shape is not 1)
  //    use f when dimension gets to dimension i.
  binary_op_recurse(m,ndim-1,shape,z,zst,x,xst,y,yst,f);
  return;
MemoryError:
  fprintf(stderr,"Error: %s(%d)\n" 
                 "\tCould not allocate block of %zu elements on stack.\n",
                 __FILE__,__LINE__,2*ndim);
  exit(1);
}

static
void scalar_ip_recurse(
    int64_t m, 
    int64_t idim, 
    size_t *shape,
    T *z,size_t *zst,
    T  a,
    scalar_ip_vec_op_t *f)
{ size_t i,oz;
  if(idim<m) 
  { 
    f(zst[m]/zst[0],z,zst[0],a);
    return;
  } 
  oz = zst[idim];
  for(i=0;i<shape[idim];++i)
    scalar_ip_recurse(m,idim-1,shape,z+oz*i,zst,a,f);
}

static
void scalar_ip_op(
    size_t ndim,
    size_t *shape,
    T* z,size_t *zst,
    T a,
    scalar_ip_vec_op_t *f)
{ 
  int64_t m;
  size_t *zshape;

  // 1. Find max index such that lower dimension indexes may be collapsed(vectorized)
  //      ostrides[j]==istrides[j] and oshape[j]==ishape[j]==shape[j] for j<i
  //    where oshape and ishape are the shapes derived from the input
  //      strides.
  if(!(zshape=alloca(sizeof(size_t)*ndim*2))) goto MemoryError;
  // stride[i]   = shape[i-1]*shape[i-2]*...*shape[0]
  // so shape[i] = stride[i+1]/stride[i]
  for(m=0;m<ndim;++m)
    zshape[m] = zst[m+1]/zst[m];
  m=0;
  for(m=0;m<ndim
      &&  shape[m]==zshape[m]
      ;++m);

  // 2. Recursively operate on outer dimension (max st shape is not 1)
  //    use f when dimension gets to dimension i.
  scalar_ip_recurse(m,ndim-1,shape,z,zst,a,f);
  return;
MemoryError:
  fprintf(stderr,"Error: %s(%d)\n" 
                 "\tCould not allocate block of %zu elements on stack.\n",
                 __FILE__,__LINE__,2*ndim);
  exit(1);
}
/******************************************************************************
*  VECTOR OPS *****************************************************************
******************************************************************************/

// zs = xs + ys
static
void vadd(size_t N, T* zs, size_t stz, T* xs, size_t stx, T* ys, size_t sty)
{ size_t i;
  for(i=0;i<N;++i)
    zs[i*stz] = xs[i*stx] + ys[i*sty];
}

void add(size_t ndim, size_t *shape, T *zs, size_t *stz, T *xs, size_t *stx, T *ys, size_t *sty) 
{ binary_op(ndim,shape,zs,stz,xs,stx,ys,sty,vadd);
}

// zs = xs - ys
static
void vsub(size_t N, T* zs, size_t stz, T* xs, size_t stx, T* ys, size_t sty)
{ size_t i;
  for(i=0;i<N;++i)
    zs[i*stz] = xs[i*stx] - ys[i*sty];
}

void sub(size_t ndim, size_t *shape, T *zs, size_t *stz, T *xs, size_t *stx, T *ys, size_t *sty) 
{ binary_op(ndim,shape,zs,stz,xs,stx,ys,sty,vsub);
}

// zs = xs - 2*ys
static 
void vsub2(size_t N, T* zs, size_t stz, T* xs, size_t stx, T* ys, size_t sty)
{ size_t i;
  for(i=0;i<N;++i)
    zs[i*stz] = xs[i*stx] - 2.0f*ys[i*sty];
}

void sub2(size_t ndim, size_t *shape, T *zs, size_t *stz, T *xs, size_t *stx, T *ys, size_t *sty) 
{ binary_op(ndim,shape,zs,stz,xs,stx,ys,sty,vsub2);
}

// zs .*= a;
static 
void vmul_scalar_ip(size_t N, T* zs, size_t stz, T a)
{ size_t i;
  for(i=0;i<N;++i) zs[i*stz] *= a;
}

void mul_ip(size_t ndim, size_t *shape, T *zs, size_t *stz, T a)
{ scalar_ip_op(ndim,shape,zs,stz,a,vmul_scalar_ip);
}

/******************************************************************************
*  SCATTER/GATHER  ************************************************************
******************************************************************************/

// Gather: zs -> [zs(even indexes) zs(odd indexes)]
/* ALGORITHM
  *  - start with two halves
  *  - divide each half in two, making quarters
  *  - recurse on quarters
  *  - swap middle two quarters
  */

/* EXAMPLE
 * 1 |1'|2 |2'|3 |3'|4 |4'|5 |5'|6 |6'|7 |7'|8 |8'
 *      X     |     X     |     X     |     X
 * 1  2 |1' 2'|3  4 |3' 4'|5  6 |5' 6'|7  8 |7' 8'
 *            X           |           X
 * 1  2  3  4 |1' 2' 3' 4'|5  6  7  8 |5' 6' 7' 8' 
 *                        X
 * 1  2  3  4  5  6  7  8 |1' 2' 3' 4' 5' 6' 7' 8'    
 */

// swaps xs and ys.  ignores sz.
static
void swap_op(size_t N, T* zs, size_t stz, T* xs, size_t stx, T* ys, size_t sty)
{ size_t i;
  
  for(i=0;i<N;++i)
  { register T t;
    T *x,*y;
    x=xs+i*stx, y=ys+i*sty;
     t=*x;
    *x=*y;
    *y=t;
  }
}

void swap(size_t ndim, size_t *shape, T *x, size_t *sx, T *y, size_t *sy)
{ binary_op(ndim,shape,x,sx,x,sx,y,sy,swap_op);
}

static
void gather_recursion(size_t idim, size_t N, size_t st, T* left, T* right, size_t ndim, size_t* shape, size_t *strides)
{ 
  size_t halfN;
  halfN=N/2;
  if(halfN)
  { T *ll,*lr,*rl,*rr;
    size_t *s;
    s = shape + idim;
    ll = left;
    lr = left+st*halfN;
    rl = right;
    rr = right+st*halfN;
    gather_recursion(idim,halfN,st,ll,lr,ndim,shape,strides);
    gather_recursion(idim,halfN,st,rl,rr,ndim,shape,strides);
    { size_t t = *s;
      *s = halfN;
      swap(ndim,shape,lr,strides,rl,strides);
      *s = t;
    }
  } 
}

void gather(size_t idim, size_t ndim, size_t *shape, T* z, size_t *st)
{ //1. transpose so idim on last dim
  size_t N,s;
  N = shape[idim];
  s = st[idim];
  gather_recursion(idim,N/2,s,z,z+s*(N/2),ndim,shape,st);
}

// Scatter: zs -> [?]
/* ALGORITHM
 * - start with two halves
 * - split each half to yield quarters
 * - swap middle two quarters 
 * - recurse on quarters
 */

/* EXAMPLE
 * 1  2  3  4  5  6  7  8 |1' 2' 3' 4' 5' 6' 7' 8'
 *                        X
 * 1  2  3  4 |1' 2' 3' 4'|5  6  7  8 |5' 6' 7' 8' 
 *            X           |           X
 * 1  2 |1' 2'|3  4 |3' 4'|5  6 |5' 6'|7  8 |7' 8'
 *      X     |     X     |     X     |     X
 * 1 |1'|2 |2'|3 |3'|4 |4'|5 |5'|6 |6'|7 |7'|8 |8'
 */

static
void scatter_recursion(size_t idim, size_t N, size_t st, T* left, T* right, size_t ndim, size_t* shape, size_t *strides)
{ 
  size_t halfN;
  halfN=N/2;
  if(halfN)
  { T *ll,*lr,*rl,*rr;
    size_t *s;
    s = shape + idim;
    ll = left;
    lr = left+st*halfN;
    rl = right;
    rr = right+st*halfN;
    { size_t t = *s;
      *s = halfN;
      swap(ndim,shape,lr,strides,rl,strides);
      *s = t;
    }
    scatter_recursion(idim,halfN,st,ll,lr,ndim,shape,strides);
    scatter_recursion(idim,halfN,st,rl,rr,ndim,shape,strides);
  } 
}

void scatter(size_t idim, size_t ndim, size_t *shape, T* z, size_t *st)
{ //1. transpose so idim on last dim
  size_t N,s;
  N = shape[idim];
  s = st[idim];
  scatter_recursion(idim,N/2,s,z,z+s*(N/2),ndim,shape,st);
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
void kernel_ip(size_t ndim, size_t *shape, T* out, size_t *s, size_t idim)
{ size_t N,off;                                                               // e.g. (matlab) - the precise indexing is determined by strides
  N=shape[idim];
  off=s[idim];
  shape[idim]=N/2;
  s[idim]=2*off;
  add (ndim, shape, out    , s, out, s, out+off, s);
  sub2(ndim, shape, out+off, s, out, s, out+off, s);
  shape[idim]=N;
  s[idim]=off;
  mul_ip(ndim, shape, out, s, INVSQRT2);
  gather(idim, ndim, shape, out, s);
}

static
void ikernel_ip(size_t ndim, size_t *shape, T* out, size_t *s, size_t idim)
{ size_t N,off;                                                                 // e.g. (matlab) - the precise indexing is determined by strides
  N=shape[idim]/2;
  off=s[idim];
  shape[idim]=N;
  add (ndim, shape, out      , s, out, s, out+off*N, s);
  sub2(ndim, shape, out+off*N, s, out, s, out+off*N, s);
  shape[idim]=N*2;
  mul_ip(ndim, shape, out, s, INVSQRT2);
  scatter(idim, ndim, shape, out, s);
}
#if 0
{ size_t halfN;                                                  // e.g. (matlab) - the precise indexing is determined by strides
  halfN = N;
  vadd (halfN,out        ,s,out,s,out+halfN*s,s);                // out(     1:N,:) =  in(1:N,:)+in((N+1):NN,:);
  vsub2(halfN,out+s*halfN,s,out,s,out+halfN*s,s);                // out((N+1):NN,:) = out(1:N,:)-2*in((N+1):NN,:);
  vmul_scalar_ip(N,out,s,INVSQRT2);
  scatter(N,out,s);
}
#endif

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
  { if(self->shapes) free(self->shapes);
    self->shapes=0;
  }
}

static
void GetDomains(DomainList* out, size_t ndim, size_t* dims)
{ unsigned n=0;
  size_t i,j;

  // compute number of iterations
  for(i=0;i<ndim;++i)
  { unsigned v = u64log2(dims[i]);
    n = (n>v)?n:v; // max(n,v);
  }

  DomainListRealloc(out,ndim,n);

  // compute domains
  size_t *cursor = out->shapes;
  for(j=0;j<ndim;++j)
    *cursor++ = dims[j]; 
  for(i=1;i<n;++i)
    for(j=0;j<ndim;++j)
    { size_t next = cursor[-ndim]>>1;
      *cursor++ = (next>1)?next:1;  //!! DOUBLE CHECK
    }
}

static inline
void DomainListResetCursor(DomainList *self)
{ self->cursor=0;
}

static inline
size_t* NextDomain(DomainList* self)
{ size_t t = self->cursor;
  size_t n = self->sz[1];
  self->cursor = (t+1)%(n+1);   //inc - one extra for end of sequence
  if(t==n) 
    return 0;
  return self->shapes + t*self->sz[0];
}

static inline
size_t* PrevDomain(DomainList* self)
{ size_t t = self->cursor;
  size_t n = self->sz[1];
  self->cursor = (t+1)%(n+1);   //inc - one extra for end of sequence
  if(t==n) 
    return 0;
  return self->shapes + (n-t-1)*self->sz[0];
}

void PrintDomains(DomainList* dl)
{ size_t* d,i;
  while((d=NextDomain(dl)))
  { printf("[ %3zu",d[0]);
    for(i=1;i<dl->sz[0];++i)
      printf(", %3zu",d[i]);
    printf(" ]\n");
  }
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

void HaarWorkspaceInit(HaarWorkspace *ws)
{ memset(ws,0,sizeof(HaarWorkspace));
}

void HaarWorkspaceClean(HaarWorkspace* ws)
{ DomainListClean(&ws->domains);
}

void haar(HaarWorkspace* ws, size_t ndim, size_t* shape, T* out, size_t* ostrides, T* in, size_t* istrides)
{ DomainList *domains;
  size_t *domain,i;

  copy(ndim,shape,out,ostrides,in,istrides);
  
  domains = &ws->domains;
  DomainListResetCursor(domains);
  GetDomains(domains,ndim,shape);
  while((domain=NextDomain(domains)))
    for(i=0;i<ndim;++i)
      if(domain[i]>1)
        kernel_ip(ndim, domain, out, ostrides, i);
        //kernel(domain[i],out,ostrides[i],in,istrides[i]);
        //kernel_ip(domain[i],out,ostrides[i]);
}

void ihaar(HaarWorkspace* ws, size_t ndim, size_t* shape, T* out, size_t* ostrides, T* in, size_t* istrides)
{ DomainList *domains;
  size_t *domain,i;
  
  copy(ndim,shape,out,ostrides,in,istrides);

  domains = &ws->domains;
  DomainListResetCursor(domains);
  GetDomains(domains,ndim,shape);
  while((domain=PrevDomain(domains)))
    for(i=0;i<ndim;++i)
      if(domain[i]>1)
        ikernel_ip(ndim, domain, out, ostrides, i);
        //ikernel(domain[i],out,ostrides[i],in,istrides[i]);
        //ikernel_ip(domain[i],out,ostrides[i]);
}
