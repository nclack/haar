/** \file
 * z-order transform
 *
 * Potential optimization
 * - get_strides ends up computing the same values over and over again...
 *   that is the get_bit(i)*strides[i] is always the same...
 *   \see DomainList
 *
 *   - for each level there are a d^ndim children
 *   - log2(shape[0]) levels
 *   - for each level need 
 *     - the shape
 *     - the input block strides
 *     - the output block strides
 *
 * - right now the terminal case is when the shape is 1x1x1x...1
 *
 *   However, stopping at the 2x2x2x...2 level, could take advantage of
 *   block copies?  The size of the block is 2xsizeof(T) so maybe not
 *   a big deal.
 *
 *   A more general view of this optimization is to unroll the last
 *   few iterations to avoid some excess computation.
 *
 * \author Nathan Clack <clackn@janelia.hhmi.org>
 * \date   2011
 */   
#include <stdint.h>
#include <stdlib.h>
#include <string.h> //for memcpy

typedef float   T;

#define MAXDIM (1024)
#define get_bit(bits,x) (((bits)>>x)&1)

/** Internally used utility function.
 *  \private
 *  \returns The byte offset to a point in the array specified by a bit vector
 *  \a bits.
 *
 *  \see zorder
 *  \see izorder
 *
 *  \param[in] ndim    The number of dimensions.
 *  \param[in] shape   The shape of the (sub)volume.
 *  \param[in] strides An \a ndim+1 length array describing the memory layout.
 *  \param[in] bits    A bit vector describing a subdivision scheme.
 */
static inline
size_t get_offset(size_t ndim, size_t *shape, size_t *strides, uint64_t bits)
{ size_t i,off=0;
  for(i=0;i<ndim;++i)
    off+=get_bit(bits,i)*strides[i]*shape[i];
  return off;
}


/** \defgroup ZOrderGroup Z-Order Transform
 *  @{
 */

/** Forward z-order transform for nD volumes.
 *
 *  \param[in] ndim      The number of dimensions.  The input and output fields must 
 *                       have the same number of dimensions, but the size of some    
 *                       dimensions may be 1.                                        
 *  \param[in] shape     An array of size \a ndim.  It describes the size of the     
 *                       rectangular volume to copy.                                 
 *  \param[in,out] out   The output data volume.                                     
 *  \param[in] ostrides  An array of size \a ndim+1 that describes the memory        
 *                       layout of \a out. \c ostrides[i] should be the number       
 *                       of items between two adjacent voxels on dimension \c        
 *                       i. \c ostrides[ndim] is the total number of bytes in        
 *                       \a out.                                                     
 *  \param[in] in        The input data volume                                       
 *  \param[in] istrides  An array of size \a ndim+1 that describes the memory        
 *                       layout of \a in.                                            
 *
 *  \see izorder
 */
void zorder(size_t ndim, size_t *shape,
            T *out, size_t *ostrides,
            T *in , size_t *istrides)
{ size_t i;
  uint64_t ichild,children,n;
  size_t tshape[MAXDIM];
  memcpy(tshape,shape,sizeof(size_t)*ndim);
  n=1;
  for(i=0;i<ndim;++i)                                        // 1. half each dimension
    n*=(tshape[i]/=2);
  children = 1<<ndim;                                        /* 2^ndim                 */
  if(tshape[0]!=0)
  { for(i=0;i<children;++i)                                  // 2. z-order children
      zorder(ndim,tshape,
          out + i*n*ostrides[0],ostrides,
          in  + get_offset(ndim,tshape,istrides,i),istrides);
    return;
  }
  else                                                       // 3. copy the blocks
  { *out = *in;
    return;
  }
}

/** Inverse z-order transform
 *
 *  \param[in] ndim      The number of dimensions.  The input and output fields must 
 *                       have the same number of dimensions, but the size of some    
 *                       dimensions may be 1.                                        
 *  \param[in] shape     An array of size \a ndim.  It describes the size of the     
 *                       rectangular volume to copy.                                 
 *  \param[in,out] out   The output data volume.                                     
 *  \param[in] ostrides  An array of size \a ndim+1 that describes the memory        
 *                       layout of \a out. \c ostrides[i] should be the number       
 *                       of items between two adjacent voxels on dimension \c        
 *                       i. \c ostrides[ndim] is the total number of bytes in        
 *                       \a out.                                                     
 *  \param[in] in        The input data volume                                       
 *  \param[in] istrides  An array of size \a ndim+1 that describes the memory        
 *                       layout of \a in.                                            
 *
 *  Implementation is the same as the forward transform except the role
 *  of out and in are swapped.
 *
 *  \see zorder
 */
void izorder(size_t ndim, size_t *shape,
             T *out, size_t *ostrides,
             T *in , size_t *istrides)
{ size_t i;
  uint64_t ichild,children,n;
  size_t tshape[MAXDIM];
  memcpy(tshape,shape,sizeof(size_t)*ndim);
  n=1;
  for(i=0;i<ndim;++i)                                        // 1. half each dimension
    n*=(tshape[i]/=2);
  children = 1<<ndim;                                        /* 2^ndim                 */
  if(tshape[0]!=0)
  { for(i=0;i<children;++i)                                  // 2. z-order children
      izorder(ndim,tshape,
          out + get_offset(ndim,tshape,ostrides,i),ostrides,
          in  + i*n*istrides[0],istrides);
    return;
  }
  else                                                       // 3. copy the blocks
  { 
    *out = *in;
    return;
  }
}
/** @} */ // end zorder transform group
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
