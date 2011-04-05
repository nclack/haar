#include <stdint.h>

typedef float   T;

#define get_bit(bits,x) (((bits)>>x)&1)

static inline
size_t get_offset(size_t ndim, size_t *shape, size_t *strides, uint64_t bits)
{ size_t i,off=0;
  for(i=0;i<ndim;++i)
    off+=get_bit(bits,i)*strides[i]*shape[i];
  return off;
}

// Potential optimization
// - get_strides ends up computing the same values over and over again...
//   that is the get_bit(i)*strides[i] is always the same...
//   
void zorder(size_t ndim, size_t *shape, T *data, size_t *strides)
{ size_t i,idim,halfN;
  uint64_t ichild,nchildren;        // limited to 64 dimensions without a bigint impl
  char index[sizoef(ichild)];
  for(i=0;i<ndim;++i)               // 1. half each dimension
    shape[idim]/=2;
  children = 1<<ndim;
  for(ichild=0;ichild<children;++i) // 2. z-order children
    zorder(ndim,shape,data+get_offset(ndim,strides,ichild),strides); 
  for(ichild=0;ichild<children;++i) // 3. copy the blocks 
    //copy
    ;


}

// tree
// each node has 2^ndim branches
// branches are enumerated by offsets, all shapes are the same
//
// for 2d
//  [z(0,0) z(0,1) z(1,0) z(1,1)]              //0,1,2,3
// for 3d
//  [z(0,0,0) z(0,0,1) z(0,1,0) z(0,1,1) ... ] //0,1,2,3,..7
//
// indexes are  the binary representation of 0 to 2^ndim-1
