/**
 * \file
 * Haar transform
 * \author Nathan Clack <clackn@janelia.hhmi.org>
 * \date   2011
 *
 * \todo Explicit optimization of vector ops for unit stride.
 * \todo Use the restrict qualifier where applicable.
 * \todo Add C API generalized over scalar types.
 * \todo Add C++ templated API.
 * \todo Explicitly in-place versions of haar() and ihaar()?
 *       This would make room for out-of-place versions that
 *       can use the \c restrict keyword.  Not sure if necessary.
 *       Or! Detect when volumes overlap in memory and use
 *       \c restrict appropriately from there.
 *       I'm pretty sure that right now, the transform will fail for volumes
 *       that overlap in memory.
 * \todo Extend to general Fast Wavelet Transform
 * \todo CUDA?
 */

/** \mainpage nD Haar transform and friends
  
    This is a bit of a personal project of mine.  I wanted to play around with
    n-dimensional wavelet transforms and z-ordering.
  
    I've got a few command line tools for executing the transforms on TIFF
    stacks.  These rely on Mylib, a library for manipulating nd arrays. 
    Mylib hasn't been released to the public yet, so you might not be
    able to use the command line utilities.
  
    The core routines are independant of Mylib though, so you can still 
    use these.
  
    I have a test suite that uses 
    <a href="http://code.google.com/p/googletest/">Google's testing
    framework</a>.
  
    \section docStartingPoints Starting points
  
    \ref pageBuilding
  
    \ref pageArray
  
    \see HaarGroup
    \see ZOrderGroup
    \see MylibGroup
  
  
    \page pageBuilding Building
    I use CMake for the build system.  You can download that 
    <a href="http://www.cmake.org/cmake/resources/software.html">here</a>.
  
    Once you have cmake installed.  Open a command prompt and navigate to 
    the base directory of the repository.  This will be the <tt>\<root\></tt>
    directory that looks like:
    \verbatim
    <root>
      CMakeLists.txt
      README
      3rdParty/
      app/
      src/
      test/
      ...
    \endverbatim
  
    Once you're there,  run these commands:
    \verbatim
    mkdir build
    cd build
    cmake ..
    \endverbatim
  
    If everthing goes correctly, you'll have a Makefile or Visual Studio
    solution inside the build directory that you can use to actually
    compile the code.
  
    To enable testing, put your copy of <a
    href="http://code.google.com/p/googletest/">googletest</a> into the \c
    3rdParty subdirectory so things look like this:
    \verbatim
    <root>
      ...
      3rdParty\
        gtest\
      ...
    \endverbatim
  
    \page pageArray nD-Array Representation
  
    The transforms here primarily operate on n-dimensional scalar fields of
    finite volume, which we'll call arrays.  In a computer's memory, each scalar
    (aka voxel or pixel) has to get laid out onto a line.  There are a few
    different ways to do this, but one common method is to use a linear
    projection like this:
    \verbatim

        offset = strides * r
    \endverbatim
    where the \c offset is the address where we will store the value at the
    n-dimensional coordinate \c r (relative to the origin), \c strides is a
    projection vector, and \c * is a dot product.
  
    A good choice for the \c strides vector is to compute it from an array's
    shape so that
    \verbatim

        strides[i+1]/strides[i] = dims[i] ; 0<i<=n+1
        strides[0]              = 1
    \endverbatim
    where \c dims is a vector with the array's shape on each of the \c n
    dimensions.

    I don't enforce a particular choice for the \c strides vector here, though
    strange choices might make for strange results.  However, I do 
    require that the lowest dimension (<tt>i=0</tt>) is contiguous in 
    memory (<tt>strides[0]=1</tt>). This allows some optimizations when copying
    subvolumes between arrays.  Also, <tt>strides[n+1]</tt> must be the total 
    number of elements in the array.

    Also, I'm not super sure what I do and don't require, so \a caveat \a lector.
 */

#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include <stdio.h>

#define SQRT2    1.41421356237309504880
#define INVSQRT2 0.70710678118654752440

typedef float T;          ///< algorithms are implemented for this scalar type
typedef int64_t stride_t; ///< pointer offset type

// UTIL ////////////////////////////////////////////////////////////////////////

/** A fast log2
  Algorithm taken from http://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
  O(log(nbits))
  Divide and conquer with table lookup over last 8 bits.
*/
unsigned u32log2(uint32_t v)
{
  static const char LogTable256[256] =
  {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
    LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
  };
#undef LT

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

/** A fast log2
  Algorithm taken from http://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
  O(log(nbits))
  Divide and conquer with table lookup over last 8 bits.
*/
unsigned u64log2(uint64_t v)
{ register uint64_t tt;
  return (tt=v>>32)?(32+u32log2((uint32_t)tt)):u32log2((uint32_t)v);
}

/** Internally used utility function.
 *  \see copy
 */
static
void copy_recurse(int64_t m, int64_t idim,stride_t *shape,T* out,stride_t *ostrides,T *in,stride_t *istrides)
{ stride_t i,os,is;
  if(idim<0) { *out = *in; return; }
  if(idim<m)
  { stride_t nelem = shape[idim];
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

/** Copy a subvolume of one n-dimensional scalar field into another scalar field.

    Attempts to optimize for contiguous blocks.  Will use memcpy() where
    appropriate for fast copies.

    When \a in == \a out, does nothing.

    This routine imposes an ordering for the strides and shape arrays.  The
    convetion used is that the lowest index should correspond to the smallest
    "natural" stride.  For example, an N-element 1D array would have stride:
    <tt>{1,N}</tt> and shape: <tt>{N}</tt>.

    \param[in] ndim      The number of dimensions.  The input and output fields must
                         have the same number of dimensions, but the size of some
                         dimensions may be 1.
    \param[in] shape     An array of size \a ndim.  It describes the size of the
                         rectangular volume to copy.
    \param[in,out] out   The output data volume.
    \param[in] ostrides  An array of size \a ndim+1 that describes the memory
                         layout of \a out. \c ostrides[i] should be the number
                         of items between two adjacent voxels on dimension \c
                         i. \c ostrides[ndim] is the total number of bytes in
                         \a out.
    \param[in] in        The input data volume
    \param[in] istrides  An array of size \a ndim+1 that describes the memory
                         layout of \a in.
 */
void copy(stride_t ndim,stride_t *shape,T* out,stride_t *ostrides,T *in,stride_t *istrides)
{
  int64_t m;
  stride_t *oshape,*ishape;
  // 0. Nothing to do?
  if(out==in) return;
  // 1. Find max index such that a block copy is ok
  //      ostrides[j]==istrides[j] and oshape[j]==ishape[j]==shape[j] for j<i
  //    where oshape and ishape are the shapes derived from the input
  //      strides.
  if(!(oshape=alloca(sizeof(stride_t)*ndim*2))) goto MemoryError;
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
  fprintf(stderr,"Could not allocate block of %lld elements on stack.",2*ndim);
  exit(1);
}

// nD Array Ops  ///////////////////////////////////////////////////////////////
/** <tt> z[:] <- x[:] op y[:]</tt>

    Performs a binary operation on two vectors \a x and \a y, assigning output to
    the vector \a z.

    Example:
    \code
    void vadd(stride_t N,
              T* zs, stride_t stz,
              T* xs, stride_t stx,
              T* ys, stride_t sty)
    { stride_t i;
      for(i=0;i<N;++i)
        zs[i*stz] = xs[i*stx] + ys[i*sty];
    }
    \endcode

    \see binary_op
    \param[in]     N     The number of elements on which to operate.
    \param[in,out] z     The output data vector.
    \param[in]     zst   The offset between successive elements in \a z.
    \param[in]     x     An input data vector.
    \param[in]     xst   The offset between successive elements in \a x.
    \param[in]     y     An input data vector.
    \param[in]     yst   The offset between successive elements in \a y.
 */
typedef void    (binary_vec_op_t)(stride_t N,T* z,stride_t zst,T* x,stride_t xst,T* y,stride_t yst);
/** <tt> z[:] .op a</tt>

    Performs an element-wise in-place operation on an input vector \a z.  The
    operation is parameterized by \a a.

    Example:
    \code
    void vmul_scalar_ip(stride_t N, T* zs, stride_t stz, T a)
    { stride_t i;
      for(i=0;i<N;++i) zs[i*stz] *= a;
    }
    \endcode
 */
typedef void (scalar_ip_vec_op_t)(stride_t,T*,stride_t,T);

/** Internally used utility function
    \see binary_op
 */
static
void binary_op_recurse(
    int64_t m,
    int64_t idim,
    stride_t *shape,
    T* z,stride_t *zst,
    T* x,stride_t *xst,
    T* y,stride_t *yst,
    binary_vec_op_t *f)
{ stride_t i,ox,oy,oz;
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

/** Applies a binary operation <tt>z=f(x,y)</tt> element-wise across
    n-dimensional scalar fields.

    Example:
    \code
    void vadd(stride_t N,
              T* zs, stride_t stz,
              T* xs, stride_t stx,
              T* ys, stride_t sty)
    { stride_t i;
      for(i=0;i<N;++i)
        zs[i*stz] = xs[i*stx] + ys[i*sty];
    }

    void add(stride_t ndim, stride_t *shape,
             T *zs, stride_t *stz,
             T *xs, stride_t *stx,
             T *ys, stride_t *sty)
    { binary_op(ndim,shape,zs,stz,xs,stx,ys,sty,vadd);
    }
    \endcode

    \param[in] ndim      The number of dimensions.  All fields must
                         have the same number of dimensions, but the size of some
                         dimensions may be 1.
    \param[in] shape     An array of size \a ndim.  It describes the size of the
                         rectangular volume on which to operate.
    \param[in,out] z     The output data volume.
    \param[in]     zst   An array of size \a ndim+1 that describes the memory
                         layout of \a z.
    \param[in]     x     An input data volume.
    \param[in]     xst   An array of size \a ndim+1 that describes the memory
                         layout of \a x.
    \param[in]     y     An input data volume.
    \param[in]     yst   An array of size \a ndim+1 that describes the memory
                         layout of \a y.
    \param[in]     f     The binary function to apply.

 */
static
void binary_op(
    stride_t ndim,
    stride_t *shape,
    T* z,stride_t *zst,
    T* x,stride_t *xst,
    T* y,stride_t *yst,
    binary_vec_op_t *f)
{
  int64_t m;
  stride_t *zshape,*xshape,*yshape;

  // 1. Find max index such that lower dimension indexes may be collapsed(vectorized)
  //      ostrides[j]==istrides[j] and oshape[j]==ishape[j]==shape[j] for j<i
  //    where oshape and ishape are the shapes derived from the input
  //      strides.
  if(!(zshape=alloca(sizeof(stride_t)*ndim*3))) goto MemoryError;
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
                 "\tCould not allocate block of %lld elements on stack.\n",
                 __FILE__,__LINE__,2*ndim);
  exit(1);
}

/** Internally used utility function. \see scalar_ip_op */
static
void scalar_ip_recurse(
    int64_t m,
    int64_t idim,
    stride_t *shape,
    T *z,stride_t *zst,
    T  a,
    scalar_ip_vec_op_t *f)
{ stride_t i,oz;
  if(idim<m)
  {
    f(zst[m]/zst[0],z,zst[0],a);
    return;
  }
  oz = zst[idim];
  for(i=0;i<shape[idim];++i)
    scalar_ip_recurse(m,idim-1,shape,z+oz*i,zst,a,f);
}

/** Applies a scalar function <tt>z=f(z;a)</tt> element-wise and in-place to an n-dimensional scalar field.

    Example:
    \code
    void vmul_scalar_ip(stride_t N, T* zs, stride_t stz, T a)
    { stride_t i;
      for(i=0;i<N;++i) zs[i*stz] *= a;
    }

    void mul_ip(stride_t ndim, stride_t *shape, T *zs, stride_t *stz, T a)
    { scalar_ip_op(ndim,shape,zs,stz,a,vmul_scalar_ip);
    }
    \endcode

    \param[in] ndim      The number of dimensions.  All fields must
                         have the same number of dimensions, but the size of some
                         dimensions may be 1.
    \param[in] shape     An array of size \a ndim.  It describes the size of the
                         rectangular volume on which to operate.
    \param[in,out] z     The data volume on which to operate.
    \param[in]     zst   An array of size \a ndim+1 that describes the memory
                         layout of \a z.
    \param[in]     a     The scalar argument
    \param[in]     f     The function to apply
 */
static
void scalar_ip_op(
    stride_t ndim,
    stride_t *shape,
    T* z,stride_t *zst,
    T a,
    scalar_ip_vec_op_t *f)
{
  int64_t m;
  stride_t *zshape;

  // 1. Find max index such that lower dimension indexes may be collapsed(vectorized)
  //      ostrides[j]==istrides[j] and oshape[j]==ishape[j]==shape[j] for j<i
  //    where oshape and ishape are the shapes derived from the input
  //      strides.
  if(!(zshape=alloca(sizeof(stride_t)*ndim*2))) goto MemoryError;
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
                 "\tCould not allocate block of %lld elements on stack.\n",
                 __FILE__,__LINE__,2*ndim);
  exit(1);
}

// VECTOR OPS //////////////////////////////////////////////////////////////////

/// 1D: zs = xs + ys
static
void vadd(stride_t N, T* zs, stride_t stz, T* xs, stride_t stx, T* ys, stride_t sty)
{ stride_t i;
  for(i=0;i<N;++i)
    zs[i*stz] = xs[i*stx] + ys[i*sty];
}

/// nD: zs = xs + ys
void add(stride_t ndim, stride_t *shape, T *zs, stride_t *stz, T *xs, stride_t *stx, T *ys, stride_t *sty)
{ binary_op(ndim,shape,zs,stz,xs,stx,ys,sty,vadd);
}

/// 1D: zs = xs - ys
static
void vsub(stride_t N, T* zs, stride_t stz, T* xs, stride_t stx, T* ys, stride_t sty)
{ stride_t i;
  for(i=0;i<N;++i)
    zs[i*stz] = xs[i*stx] - ys[i*sty];
}

/// nD: zs = xs - ys
void sub(stride_t ndim, stride_t *shape, T *zs, stride_t *stz, T *xs, stride_t *stx, T *ys, stride_t *sty)
{ binary_op(ndim,shape,zs,stz,xs,stx,ys,sty,vsub);
}

/// 1D: zs = xs - 2*ys
static
void vsub2(stride_t N, T* zs, stride_t stz, T* xs, stride_t stx, T* ys, stride_t sty)
{ stride_t i;
  for(i=0;i<N;++i)
    zs[i*stz] = xs[i*stx] - 2.0f*ys[i*sty];
}

/// nD: zs = xs - 2*ys
void sub2(stride_t ndim, stride_t *shape, T *zs, stride_t *stz, T *xs, stride_t *stx, T *ys, stride_t *sty)
{ binary_op(ndim,shape,zs,stz,xs,stx,ys,sty,vsub2);
}

/// 1D: zs .*= a
static
void vmul_scalar_ip(stride_t N, T* zs, stride_t stz, T a)
{ stride_t i;
  for(i=0;i<N;++i) zs[i*stz] *= a;
}

/// nD: zs .*= a
void mul_ip(stride_t ndim, stride_t *shape, T *zs, stride_t *stz, T a)
{ scalar_ip_op(ndim,shape,zs,stz,a,vmul_scalar_ip);
}

/// 1D: swaps xs and ys.  ignores sz.
static
void swap_op(stride_t N, T* zs, stride_t stz, T* xs, stride_t stx, T* ys, stride_t sty)
{ stride_t i;

  for(i=0;i<N;++i)
  { register T t;
    T *x,*y;
    x=xs+i*stx, y=ys+i*sty;
     t=*x;
    *x=*y;
    *y=t;
  }
}

/// nD: swaps xs and ys.
void swap(stride_t ndim, stride_t *shape, T *x, stride_t *sx, T *y, stride_t *sy)
{ binary_op(ndim,shape,x,sx,x,sx,y,sy,swap_op);
}

//  SCATTER/GATHER  ////////////////////////////////////////////////////////////


/// Internally used utility function. \see gather
static
void gather_recursion(stride_t idim, stride_t N, stride_t st, T* left, T* right, stride_t ndim, stride_t* shape, stride_t *strides)
{
  stride_t halfN;
  halfN=N/2;
  if(halfN)
  { T *ll,*lr,*rl,*rr;
    stride_t *s;
    s = shape + idim;
    ll = left;
    lr = left+st*halfN;
    rl = right;
    rr = right+st*halfN;
    gather_recursion(idim,halfN,st,ll,lr,ndim,shape,strides);
    gather_recursion(idim,halfN,st,rl,rr,ndim,shape,strides);
    { stride_t t = *s;
      *s = halfN;
      swap(ndim,shape,lr,strides,rl,strides);
      *s = t;
    }
  }
}

/** Gather: <tt>zs -> [zs(even indexes) zs(odd indexes)]</tt>

   Algorithm:
     - start with two halves
     - divide each half in two, making quarters
     - recurse on quarters
     - swap middle two quarters

   To illustrate:
   \verbatim
   1 |1'|2 |2'|3 |3'|4 |4'|5 |5'|6 |6'|7 |7'|8 |8'
        X     |     X     |     X     |     X
   1  2 |1' 2'|3  4 |3' 4'|5  6 |5' 6'|7  8 |7' 8'
              X           |           X
   1  2  3  4 |1' 2' 3' 4'|5  6  7  8 |5' 6' 7' 8'
                          X
   1  2  3  4  5  6  7  8 |1' 2' 3' 4' 5' 6' 7' 8'
   \endverbatim

   Inverse of scatter().

*/
void gather(stride_t idim, stride_t ndim, stride_t *shape, T* z, stride_t *st)
{ //1. transpose so idim on last dim
  stride_t N,s;
  N = shape[idim];
  s = st[idim];
  gather_recursion(idim,N/2,s,z,z+s*(N/2),ndim,shape,st);
}


/// Internally used utility function. \see scatter
static
void scatter_recursion(stride_t idim, stride_t N, stride_t st, T* left, T* right, stride_t ndim, stride_t* shape, stride_t *strides)
{
  stride_t halfN;
  halfN=N/2;
  if(halfN)
  { T *ll,*lr,*rl,*rr;
    stride_t *s;
    s = shape + idim;
    ll = left;
    lr = left+st*halfN;
    rl = right;
    rr = right+st*halfN;
    { stride_t t = *s;
      *s = halfN;
      swap(ndim,shape,lr,strides,rl,strides);
      *s = t;
    }
    scatter_recursion(idim,halfN,st,ll,lr,ndim,shape,strides);
    scatter_recursion(idim,halfN,st,rl,rr,ndim,shape,strides);
  }
}

/** Inverse of gather().

   Algorithm:
   - start with two halves
   - split each half to yield quarters
   - swap middle two quarters
   - recurse on quarters

   To illustrate:
   \verbatim
   1  2  3  4  5  6  7  8 |1' 2' 3' 4' 5' 6' 7' 8'
                          X
   1  2  3  4 |1' 2' 3' 4'|5  6  7  8 |5' 6' 7' 8'
              X           |           X
   1  2 |1' 2'|3  4 |3' 4'|5  6 |5' 6'|7  8 |7' 8'
        X     |     X     |     X     |     X
   1 |1'|2 |2'|3 |3'|4 |4'|5 |5'|6 |6'|7 |7'|8 |8'
   \endverbatim
 */
void scatter(stride_t idim, stride_t ndim, stride_t *shape, T* z, stride_t *st)
{ //1. transpose so idim on last dim
  stride_t N,s;
  N = shape[idim];
  s = st[idim];
  scatter_recursion(idim,N/2,s,z,z+s*(N/2),ndim,shape,st);
}

// KERNEL  /////////////////////////////////////////////////////////////////////

// haar kernel: in(1:N) --> out(1:N)
// ---------------------------------
// 'out' and 'in' must not overlap
// 'st*' are the strides
//
// The matrix multiplication underlying this is self-inverse so
// the kernels end up looking very similar!

/** In-place forward Haar transform kernel.
    \param[in] ndim      The number of dimensions.
    \param[in] shape     An array of size \a ndim.  It describes the size of the
                         rectangular volume on which to operate.
    \param[in,out] out   The data volume on which to operate.
    \param[in]     s     An array of size \a ndim+1 that describes the memory
                         layout of \a out.
    \param[in]     idim  The dimension on which to operate.

    Inverse kernel is ikernel_ip().

    \see haar
 */
static
void kernel_ip(stride_t ndim, stride_t *shape, T* out, stride_t *s, stride_t idim)
{ stride_t N,off;                                                               // e.g. (matlab) - the precise indexing is determined by strides
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

/** In-place inverse Haar transform kernel.
    \param[in] ndim      The number of dimensions.
    \param[in] shape     An array of size \a ndim.  It describes the size of the
                         rectangular volume on which to operate.
    \param[in,out] out   The data volume on which to operate.
    \param[in]     s     An array of size \a ndim+1 that describes the memory
                         layout of \a out.
    \param[in]     idim  The dimension on which to operate.

    Forward kernel is kernel_ip().

    \see ihaar
 */
static
void ikernel_ip(stride_t ndim, stride_t *shape, T* out, stride_t *s, stride_t idim)
{ stride_t N,off;                                                                 // e.g. (matlab) - the precise indexing is determined by strides
  N=shape[idim]/2;
  off=s[idim];
  shape[idim]=N;
  add (ndim, shape, out      , s, out, s, out+off*N, s);
  sub2(ndim, shape, out+off*N, s, out, s, out+off*N, s);
  shape[idim]=N*2;
  mul_ip(ndim, shape, out, s, INVSQRT2);
  scatter(idim, ndim, shape, out, s);
}

//  DOMAINS  ///////////////////////////////////////////////////////////////////

/** \defgroup DomainListGroup Domain List (private)
 *  The private interface to the DomainList object
 *
 *  \see HaarGroup
 *
 *  This structure encapsulates the generation and iteration over subvolumes.
 *  A small amount of memory,<tt>O(ndims*log(max(dims)))</tt>, is used to
 *  record the subdivision plan.
 *
 *  Both the Haar and Z-Order transforms are implemented by applying some
 *  transform kernel to successive subvolumes of the data.  The inverses
 *  operate on the same subvolumes, but in reverse order.
 *
 *  This enables a fairly concise implementation of the transforms.
 *
 *  Example:
 *  Input - \c ndim,\c *dims
 *  \code
 *  { DomainList dl;
 *    memset(dl,0,sizeof(dl));
 *    GetDomains(&dl,ndim,dims);
 *    // Use result
 *    DomainListClean(&dl);
 *  }
 *  \endcode
 *  @{
 */

/** 
 *  Keeps track of sub-volumes for algorithms that rely on
 *  recursive subdivision.
 *
 *  \see DomainListRealloc
 *  \see DomainListClean
 *  \see GetDomains
 *  \see DomainListResetCursor
 *  \see NextDomain
 *  \see PrevDomain
 *  \see PrintDomains
 */
typedef struct tagDomainList
{ stride_t *shapes; ///< Resizable 2d array of shapes (n x ndim)
  stride_t sz[2];   ///< Shape of the \a shapes array
  stride_t cursor;  ///< Tracks the current iteration point.
} DomainList;

/** Allocates/Reallocates internal storage as necessary.
 *  \see GetDomains
 */
static
void DomainListRealloc(DomainList *self, stride_t ndim, stride_t n)
{ if(self->shapes)
    self->shapes = realloc(self->shapes,sizeof(stride_t)*ndim*n);
  else
    self->shapes = malloc(sizeof(stride_t)*ndim*n);
  self->sz[1]=n;    //rows
  self->sz[0]=ndim; //cols
}

/// Releases allocated resources.
static
void DomainListClean(DomainList *self)
{ if(self)
  { if(self->shapes) free(self->shapes);
    self->shapes=0;
  }
}

/// Computes a subdivision plan.
static
void GetDomains(DomainList* out, stride_t ndim, stride_t* dims)
{ unsigned n=0;
  stride_t i,j;

  // compute number of iterations
  for(i=0;i<ndim;++i)
  { unsigned v = u64log2(dims[i]);
    n = (n>v)?n:v; // max(n,v);
  }

  DomainListRealloc(out,ndim,n);

  // compute domains
  stride_t *cursor = out->shapes;
  for(j=0;j<ndim;++j)
    *cursor++ = dims[j];
  for(i=1;i<n;++i)
    for(j=0;j<ndim;++j)
    { stride_t next = cursor[-ndim]>>1;
      *cursor++ = (next>1)?next:1;
    }
}

/// Sets the iteration point back to the beginning.
static inline
void DomainListResetCursor(DomainList *self)
{ self->cursor=0;
}

/// \returns The shape of the next subdivision, or NULL if there is none.
static inline
stride_t* NextDomain(DomainList* self)
{ stride_t t = self->cursor;
  stride_t n = self->sz[1];
  self->cursor = (t+1)%(n+1);   //inc - one extra for end of sequence
  if(t==n)
    return 0;
  return self->shapes + t*self->sz[0];
}

/// \returns The shape of the previous subdivision, or NULL if there is none.
static inline
stride_t* PrevDomain(DomainList* self)
{ stride_t t = self->cursor;
  stride_t n = self->sz[1];
  self->cursor = (t+1)%(n+1);   //inc - one extra for end of sequence
  if(t==n)
    return 0;
  return self->shapes + (n-t-1)*self->sz[0];
}

/// Prints the sub-division plan (for debugging).
void PrintDomains(DomainList* dl)
{ stride_t* d,i;
  while((d=NextDomain(dl)))
  { printf("[ %3lld",d[0]);
    for(i=1;i<dl->sz[0];++i)
      printf(", %3lld",d[i]);
    printf(" ]\n");
  }
}
/** @} */ //end of DomainListGroup

// TRANSFORM  //////////////////////////////////////////////////////////////////

/** \defgroup HaarGroup Haar
 *  @{
 */

/** The HaarWorkspace object keeps track of resources needed to compute the
 *  haar() transform or it's inverse ihaar().
 *
 *  It's meant to be reused.  If one is making several successive calls to
 *  haar() or ihaar() in the same thread, it's ok to use the same workspace for
 *  each call.  This recycles internally allocated resources.
 *
 *  \see DomainListGroup
 */
typedef struct tagHaarWorkspace
{
  DomainList domains; ///< The subdivision plan.
} HaarWorkspace;

/** Initializes the workspace.
 *
 *  Just fills the struct with zeros.
 */
void HaarWorkspaceInit(HaarWorkspace *ws)
{ memset(ws,0,sizeof(HaarWorkspace));
}

/// Releases internally allocated resources. 
void HaarWorkspaceClean(HaarWorkspace* ws)
{ DomainListClean(&ws->domains);
}

/** Performs the forward Haar transform.
 *
 * When the input and output array are the same, the transform is
 * in place.  Otherwise, the input subvolume is copied to the 
 * output subvolume, and the transform is done in-place on the output.
 *
 * A complicated example:
 *
 * The input volume is a 100x200x300 (rows x cols x planes) array.
 * The (preallocated) output array is 100x100x100.  Perform the transform on a
 * 50x50x50 subvolume and center the result in the output array.
 *
 * \code
 * HaarWorkspace ws;
 * stride_t istride[] = {1,200,100*200,100*200*300},
 *          ostride[] = {1,100,100*100,100*100,100},
 *          shape[]   = {50,50,50};
 * HaarWorkspaceInit(ws);
 * haar(&ws,shape,
 *      out+25*100*100+25*100+25, // put transform result at (25,25,25) in
 *      ostride,in,instride);     // the output array
 * HaarWorkspaceClean(&ws);
 * \endcode
 *
 * \param[in] ws        A HaarWorkspace object.
 * \param[in] ndim      The number of dimensions.  The input and output fields must
 *                      have the same number of dimensions, but the size of some
 *                      dimensions may be 1.
 * \param[in] shape     An array of size \a ndim.  It describes the size of the
 *                      rectangular volume to copy.
 * \param[in,out] out   The output data volume.
 * \param[in] ostrides  An array of size \a ndim+1 that describes the memory
 *                      layout of \a out. \c ostrides[i] should be the number
 *                      of items between two adjacent voxels on dimension \c
 *                      i. \c ostrides[ndim] is the total number of bytes in
 *                      \a out.
 * \param[in] in        The input data volume
 * \param[in] istrides  An array of size \a ndim+1 that describes the memory
 *                      layout of \a in.
 */
void haar(HaarWorkspace* ws, stride_t ndim, stride_t* shape, T* out, stride_t* ostrides, T* in, stride_t* istrides)
{ DomainList *domains;
  stride_t *domain,i;

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

/** Performs the inverse Haar transform.
 *  \see haar
 *
 * When the input and output array are the same, the transform is
 * in place.  Otherwise, the input subvolume is copied to the 
 * output subvolume, and the transform is done in-place on the output.
 *
 * \param[in] ws        A HaarWorkspace object.
 * \param[in] ndim      The number of dimensions.  The input and output fields must
 *                      have the same number of dimensions, but the size of some
 *                      dimensions may be 1.
 * \param[in] shape     An array of size \a ndim.  It describes the size of the
 *                      rectangular volume to copy.
 * \param[in,out] out   The output data volume.
 * \param[in] ostrides  An array of size \a ndim+1 that describes the memory
 *                      layout of \a out. \c ostrides[i] should be the number
 *                      of items between two adjacent voxels on dimension \c
 *                      i. \c ostrides[ndim] is the total number of bytes in
 *                      \a out.
 * \param[in] in        The input data volume
 * \param[in] istrides  An array of size \a ndim+1 that describes the memory
 *                      layout of \a in.
 */
void ihaar(HaarWorkspace* ws, stride_t ndim, stride_t* shape, T* out, stride_t* ostrides, T* in, stride_t* istrides)
{ DomainList *domains;
  stride_t *domain,i;

  copy(ndim,shape,out,ostrides,in,istrides); // does nothing if in == out

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

/** @}*/ //end the HaarWorkspaceGroup
